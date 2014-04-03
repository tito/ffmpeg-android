'''
Python library for decoding video
=================================

References:
    http://bambuser.com/opensource
    https://github.com/tranx/pyffmpeg/blob/master/pyffmpeg.pyx
    http://dranger.com/ffmpeg/

Some note from debugging stuff:

    #. When we are doing thread, since we are in python, force the GIL to be
    acquired by adding "with gil" at the end of the callback/main thread func.

    #. On android, seem that --embed on cython don't do threads initialization.
    __PYX_FORCE_INIT_THREADS missing ?

TODO:

    - handle error case (looping on schedule/refresh even when no video)
    - fix seek case, doesn't work when seek back.

'''

include '_ffmpeg.pxi'

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport log, exp, fabs
from cpython.string cimport PyString_FromStringAndSize
import cython

from os.path import dirname, join

cdef int g_have_register = 0
cdef int g_have_register_audio = 0
cdef AVPacket flush_pkt

cdef struct PacketQueue:
    AVPacketList *first_pkt
    AVPacketList *last_pkt
    int nb_packets
    int size
    SDL_mutex *mutex
    SDL_cond *cond
    int quit

cdef struct VideoPicture:
    int width
    int height
    int allocated
    double pts
    AVFrame *bmp
    unsigned char *ff_data
    unsigned ff_data_size

#
# Taken from tutorial 8
#

DEF AUDIO_S16SYS = 0x8010
DEF SDL_AUDIO_BUFFER_SIZE           = 1024
DEF MAX_AUDIOQ_SIZE                 = (5 * 16 * 1024)
DEF MAX_VIDEOQ_SIZE                 = (5 * 256 * 1024)
DEF AV_SYNC_THRESHOLD               = 0.01
DEF AV_NOSYNC_THRESHOLD             = 10.0
DEF SAMPLE_CORRECTION_PERCENT_MAX   = 10
DEF AUDIO_DIFF_AVG_NB               = 20
DEF VIDEO_PICTURE_QUEUE_SIZE        = 1
DEF AVCODEC_MAX_AUDIO_FRAME_SIZE    = 192000 # ffmpeg
cdef uint64_t AV_NOPTS_VALUE = 0x8000000000000000


DEF AV_SYNC_AUDIO_MASTER            = 0
DEF AV_SYNC_VIDEO_MASTER            = 1
DEF AV_SYNC_EXTERNAL_MASTER         = 2

DEF DEFAULT_AV_SYNC_TYPE            = AV_SYNC_VIDEO_MASTER
DEF FF_ALLOC_EVENT                  = 1
DEF FF_REFRESH_EVENT                = 2
DEF FF_QUIT_EVENT                   = 3
DEF FF_SCHEDULE_EVENT               = 4

cdef uint64_t global_video_pkt_pts = AV_NOPTS_VALUE

ctypedef void (*event_callback_t)(void *)

cdef struct Event:
    int name
    void *userdata
    int delay
    event_callback_t callback
    Event *next

cdef struct EventQueue:
    Event *first
    Event *last
    SDL_mutex *mutex

cdef struct VideoState:
    uint8_t         *audio_buf
    uint8_t         silence_buf[1024]
    AVFormatContext *pFormatCtx
    int             videoStream
    int             audioStream

    int             av_sync_type
    double          external_clock # external clock base
    int64_t         external_clock_time
    int             seek_req
    double          seek_percent
    double          audio_clock
    AVStream        *audio_st
    PacketQueue     audioq
    unsigned int    audio_buf_size
    unsigned int    audio_buf_index
    AVPacket        audio_pkt
    uint8_t         *audio_pkt_data
    int             audio_pkt_size
    int             audio_hw_buf_size
    double          audio_diff_cum # used for AV difference average computation
    double          audio_diff_avg_coef
    double          audio_diff_threshold
    int             audio_diff_avg_count
    double          frame_timer
    double          frame_last_pts
    double          frame_last_delay
    double          video_clock # <pts of last decoded frame / predicted pts of next decoded frame
    double          video_current_pts #<current displayed pts (different from video_clock if frame fifos are used)
    int64_t         video_current_pts_time #<time (av_gettime) at which we updated video_current_pts - used to have running video pts
    AVStream        *video_st
    PacketQueue     videoq

    VideoPicture    pictq[VIDEO_PICTURE_QUEUE_SIZE]
    int             pictq_size, pictq_rindex, pictq_windex
    SDL_mutex       *pictq_mutex
    SDL_cond        *pictq_cond
    SDL_Thread      *parse_tid
    SDL_Thread      *video_tid
    char            filename[1024]
    int             quit
    EventQueue      eq
    SwsContext      *img_convert_ctx
    Mix_Chunk       *audio_chunk
    int             audio_channel
    AVAudioResampleContext *avr

    int resample_sample_fmt
    uint64_t resample_channel_layout
    int resample_sample_rate

cdef inline int get_bits_per_sample_fmt(int fmt) nogil:
    if fmt == 0: #U8
        return 8
    elif fmt == 1: #U16
        return 16
    elif fmt == 2: #U32
        return 32
    elif fmt == 3: #FLOAT
        return sizeof(float)
    elif fmt == 4: #DOUBLE
        return sizeof(double)
    return -1

cdef inline int imin(int a, int b) nogil:
    return a if a < b else b

cdef inline double dmax(double a, double b) nogil:
    return a if a > b else b

cdef VideoState *global_video_state = NULL
cdef SDL_mutex *g_ffmpeg_mutex = NULL

#
# Mixer
# We need a way to mix several raw channels together.
# The only way was to play an empty sound on each channel, but replace the
# stream by the sound as a sound effect.
#


DEF MIX_CHANNELS_MAX = 32
cdef int mix_rate = 44100
cdef int mix_channels = 2
cdef int mix_audio_have_init = 0
cdef int mix_channels_usage[MIX_CHANNELS_MAX]
cdef int mix_sample_fmt = AV_SAMPLE_FMT_S16
cdef int mix_channel_layout = AV_CH_LAYOUT_STEREO

# manually crafted a WAV 16bit stereo of 0.1s silence
cdef str silence_wav = (
    'RIFF\x0cE\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00D\xac\x00\x00'
    '\x10\xb1\x02\x00\x04\x00\x10\x00data\xe8D\x00\x00') + '\x00' * 17640

cdef int mix_audio_init():
    global g_have_register_audio
    global mix_rate, mix_channels
    if g_have_register_audio == 1:
        return 0
    g_have_register_audio = 1

    if SDL_Init(SDL_INIT_AUDIO) < 0:
        print 'SDL_Init: %s' % SDL_GetError()
        return -1

    if Mix_OpenAudio(mix_rate, AUDIO_S16SYS, mix_channels, 1024):
        print 'Mix_OpenAudio: %s' % SDL_GetError()
        return -1

    memset(mix_channels_usage, 0, sizeof(int) * MIX_CHANNELS_MAX)

    SDL_LockAudio()

    print 'FFMPEG: Audio ask for', mix_rate, mix_channels
    Mix_QuerySpec(&mix_rate, NULL, &mix_channels)
    print 'FFMPEG: Audio ask got', mix_rate, mix_channels

    Mix_AllocateChannels(MIX_CHANNELS_MAX)

    SDL_UnlockAudio()

    return 0

cdef int ffmpeg_mutex_mgr(void **_mutex, AVLockOp op) nogil:
    cdef SDL_mutex *mutex = <SDL_mutex *>_mutex[0]
    if op == AV_LOCK_CREATE:
        mutex = <SDL_mutex *>SDL_CreateMutex()
        _mutex[0] = <void *>mutex
    elif op == AV_LOCK_OBTAIN:
        SDL_LockMutex(mutex)
    elif op == AV_LOCK_RELEASE:
        SDL_UnlockMutex(mutex)
    elif op == AV_LOCK_DESTROY:
        SDL_DestroyMutex(mutex)
        _mutex[0] = NULL
    else:
        return -1
    return 0

cdef void ffmpeg_log_callback(void *ptr, int level, const_char_ptr fmt, va_list
        vl) nogil:
    cdef char message[1024]
    cdef int type[2]
    cdef int print_prefix = 1
    av_log_format_line(ptr, level, fmt, vl, message, 1024, &print_prefix)
    # XXX if we acquire the gil here, we are still having the ffmpeg mutex, so
    # we have a deadlock
    #with gil:
    #    print 'ffmpeg:', message

cdef void ffmpeg_ensure_init():
    # ensure that ffmpeg have been registered first
    global g_have_register
    if g_have_register == 1:
        return

    g_have_register = 1

    PyEval_InitThreads()
    av_register_all()

    # ensure log will be printed
    #av_log_set_callback(ffmpeg_log_callback)

    # add mutex management
    av_lockmgr_register(ffmpeg_mutex_mgr)

    # Init audio
    mix_audio_init()


#
# User event queue, to communicate between thread and python class
# No python used, to be able at some time to remove GIL usage.
#

cdef void event_queue_init(EventQueue *q) nogil:
    memset(q, 0, sizeof(EventQueue))
    q.mutex = SDL_CreateMutex()

cdef Event *event_create() nogil:
    cdef Event *event = <Event *>malloc(sizeof(Event))
    memset(event, 0, sizeof(Event))
    return event

cdef void event_queue_put(EventQueue *q, Event *e) nogil:
    SDL_LockMutex(q.mutex)
    if q.last != NULL:
        q.last.next = e
    q.last = e
    if q.first == NULL:
        q.first = e
    SDL_UnlockMutex(q.mutex)

cdef void event_queue_put_fast(EventQueue *q, int name, void *userdata) nogil:
    cdef Event *e = event_create()
    e.name = name
    e.userdata = userdata
    event_queue_put(q, e)

cdef Event *event_queue_get(EventQueue *q) nogil:
    cdef Event *e = NULL
    with nogil:
        SDL_LockMutex(q.mutex)
    if q.first != NULL:
        e = q.first
        q.first = q.first.next
    if q.first == NULL:
        q.last = NULL
    with nogil:
        SDL_UnlockMutex(q.mutex)
    return e

cdef void event_queue_clean(EventQueue *q) nogil:
    cdef Event *e, *e2
    SDL_LockMutex(q.mutex)
    e = q.first
    while e != NULL:
        q.first = e.next
        free(e)
        e = q.first
    SDL_UnlockMutex(q.mutex)
    SDL_DestroyMutex(q.mutex)



#
# Packet Queue
#

cdef void packet_queue_init(PacketQueue *q) nogil:
    memset(q, 0, sizeof(PacketQueue))
    q.mutex = SDL_CreateMutex()
    q.cond = SDL_CreateCond()

cdef void packet_queue_clean(PacketQueue *q) nogil:
    q.quit = 1
    packet_queue_flush(q)
    SDL_LockMutex(q.mutex)
    SDL_CondSignal(q.cond)
    SDL_UnlockMutex(q.mutex)
    SDL_Delay(10)
    SDL_LockMutex(q.mutex)
    SDL_UnlockMutex(q.mutex)
    SDL_DestroyMutex(q.mutex)
    SDL_DestroyCond(q.cond)
    q.mutex = q.cond = NULL

cdef int packet_queue_put(PacketQueue *q, AVPacket *pkt) nogil:
    cdef AVPacketList *pkt1

    if av_dup_packet(pkt) < 0:
        return -1

    pkt1 = <AVPacketList *>av_malloc(sizeof(AVPacketList))
    if pkt1 == NULL:
        return - 1

    memcpy(&pkt1.pkt, pkt, sizeof(AVPacket))
    pkt1.next = NULL

    SDL_LockMutex(q.mutex)

    if q.last_pkt == NULL:
        q.first_pkt = pkt1
    else:
        q.last_pkt.next = pkt1
    q.last_pkt = pkt1
    q.nb_packets += 1
    q.size += pkt1.pkt.size

    SDL_CondSignal(q.cond)
    SDL_UnlockMutex(q.mutex)

    return 0


cdef int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block) nogil:
    cdef AVPacketList *pkt1
    cdef int ret = -1

    SDL_LockMutex(q.mutex)

    while True:

        if q.quit:
            ret = -1
            break

        pkt1 = q.first_pkt
        if pkt1 != NULL:
            q.first_pkt = pkt1.next
            if q.first_pkt == NULL:
                q.last_pkt = NULL
            q.nb_packets -= 1
            q.size -= pkt1.pkt.size
            memcpy(pkt, &pkt1.pkt, sizeof(AVPacket))
            av_freep(&pkt1)
            ret = 1
            break
        elif block == 0:
            ret = 0
            break
        else:
            SDL_CondWait(q.cond, q.mutex)

    SDL_UnlockMutex(q.mutex)

    return ret


cdef void packet_queue_flush(PacketQueue *q) nogil:
    cdef AVPacketList *pkt, *pkt1

    SDL_LockMutex(q.mutex)
    pkt = q.first_pkt
    while pkt != NULL:
        pkt1 = pkt.next
        av_free_packet(&pkt.pkt)
        av_freep(&pkt)
        pkt = pkt1
    q.last_pkt = NULL
    q.first_pkt = NULL
    q.nb_packets = 0
    q.size = 0
    SDL_UnlockMutex(q.mutex)


@cython.cdivision(True)
cdef double get_audio_clock(VideoState *vs) nogil:
    cdef double pts
    cdef int hw_buf_size, bytes_per_sec, n

    pts = vs.audio_clock
    hw_buf_size = vs.audio_buf_size - vs.audio_buf_index
    bytes_per_sec = 0
    n = vs.audio_st.codec.channels * 2
    if vs.audio_st:
        bytes_per_sec = vs.audio_st.codec.sample_rate * n
    if bytes_per_sec:
        pts -= <double>hw_buf_size / bytes_per_sec
    return pts

cdef double get_video_clock(VideoState *vs) nogil:
    cdef double delta
    delta = (av_gettime() - vs.video_current_pts_time) / 1000000.0
    return vs.video_current_pts + delta

cdef double get_external_clock(VideoState *vs) nogil:
    return av_gettime() / 1000000.0

cdef double get_master_clock(VideoState *vs) nogil:
    if vs.av_sync_type == AV_SYNC_VIDEO_MASTER:
        return get_video_clock(vs)
    elif vs.av_sync_type == AV_SYNC_AUDIO_MASTER:
        return get_audio_clock(vs)
    else:
        return get_external_clock(vs)


cdef int synchronize_audio(VideoState *vs, short *samples, int samples_size,
        double pts) nogil:
    '''Add or subtract samples to get a better sync, return new
    audio buffer size'''
    cdef int n
    cdef double ref_clock
    cdef double diff, avg_diff
    cdef int wanted_size, min_size, max_size, nb_samples
    cdef uint8_t *samples_end, *q
    cdef int nb

    n = 2 * vs.audio_st.codec.channels

    if vs.av_sync_type != AV_SYNC_AUDIO_MASTER:

        ref_clock = get_master_clock(vs)
        diff = get_audio_clock(vs) - ref_clock
        if diff < AV_NOSYNC_THRESHOLD:
            # accumulate the diffs
            vs.audio_diff_cum = diff + vs.audio_diff_avg_coef * vs.audio_diff_cum
            if vs.audio_diff_avg_count < AUDIO_DIFF_AVG_NB:
                vs.audio_diff_avg_count += 1
            else:
                avg_diff = vs.audio_diff_cum * (1.0 - vs.audio_diff_avg_coef)
                if fabs(avg_diff) >= vs.audio_diff_threshold:
                    wanted_size = samples_size + (<int>(diff * vs.audio_st.codec.sample_rate) * n)
                    min_size = samples_size * ((100 - SAMPLE_CORRECTION_PERCENT_MAX) / 100)
                    max_size = samples_size * ((100 + SAMPLE_CORRECTION_PERCENT_MAX) / 100)
                    if wanted_size < min_size:
                        wanted_size = min_size
                    elif wanted_size > max_size:
                        wanted_size = max_size

                    if wanted_size < samples_size:
                        # remove samples
                        samples_size = wanted_size
                    elif(wanted_size > samples_size):
                        # add samples by copying final sample
                        nb = (samples_size - wanted_size)
                        samples_end = <uint8_t *>samples + samples_size - n
                        q = samples_end + n
                        while nb > 0:
                            memcpy(q, samples_end, n)
                            q += n
                            nb -= n
                        samples_size = wanted_size
        else:
            # difference is TOO big reset diff stuff
            vs.audio_diff_avg_count = 0
            vs.audio_diff_cum = 0

    return samples_size


@cython.cdivision(True)
cdef int audio_decode_frame(VideoState *vs, double *pts_ptr) nogil:
    cdef:
        AVPacket *pkt = &vs.audio_pkt
        int len1, data_size, n
        double pts
        int got_frame = 0
        AVFrame *frame = NULL
        void *tmp_out
        int out_samples, out_size, out_linesize, osize, nb_samples
        int audio_resample, resample_changed


    while True:

        while vs.audio_pkt_size > 0:

            got_frame = 0

            if frame == NULL:
                frame = avcodec_alloc_frame()
                if frame == NULL:
                    return -1
            else:
                avcodec_get_frame_defaults(frame)

            len1 = avcodec_decode_audio4(vs.audio_st.codec,
                    frame, &got_frame, pkt)

            if len1 < 0:
                # if error, skip frame
                vs.audio_pkt_size = 0
                break

            vs.audio_pkt_data += len1
            vs.audio_pkt_size -= len1
            if not got_frame:
                # No data yet, get more frames
                continue

            # notes
            # dec->channels                 vs.audio_st.codec.channels
            # is->frame->format             frame.format
            # is->frame->channel_layout     frame.channel_layout
            # is->frame->sample_rate        frame.sample_rate
            # is->frame->nb_samples         frame.nb_samples

            data_size = av_samples_get_buffer_size(NULL,
                    vs.audio_st.codec.channels,
                    frame.nb_samples,
                    <AVSampleFormat>frame.format, 1)

            audio_resample = frame.format != mix_sample_fmt or \
                    frame.channel_layout != mix_channel_layout or \
                    frame.sample_rate != mix_rate
            resample_changed = frame.format != vs.resample_sample_fmt or \
                    frame.channel_layout != vs.resample_channel_layout or \
                    frame.sample_rate != vs.resample_sample_rate

            if (not vs.avr and audio_resample) or resample_changed:
                #with gil:
                #    print 'resampling audio needed, open it'
                if vs.avr:
                    avresample_free(&vs.avr)
                vs.avr = avresample_alloc_context()
                if vs.avr == NULL:
                    #with gil:
                    #    print 'Audio need resample, but unable to create avr'
                    avcodec_free_frame(&frame)
                    return -1

                #with gil:
                #    print 'in_channel_layout', frame.channel_layout
                #    print 'out_channel_layout', mix_channel_layout
                #    print 'in_sample_fmt', frame.format
                #    print 'out_sample_fmt', mix_sample_fmt
                #    print 'in_sample_rate', frame.sample_rate
                #    print 'out_sample_rate', mix_rate

                av_opt_set_int(vs.avr, 'in_channel_layout',
                        frame.channel_layout, 0)
                av_opt_set_int(vs.avr, 'out_channel_layout',
                        mix_channel_layout, 0)
                av_opt_set_int(vs.avr, 'in_sample_rate', frame.sample_rate, 0)
                av_opt_set_int(vs.avr, 'out_sample_rate', mix_rate, 0)
                av_opt_set_int(vs.avr, 'in_sample_fmt', frame.format, 0)
                av_opt_set_int(vs.avr, 'out_sample_fmt', mix_sample_fmt, 0)
                #av_opt_set_int(vs.avr, 'internal_sample_fmt', AV_SAMPLE_FMT_S16P, 0)

                ret = avresample_open(vs.avr)
                if ret != 0:
                    #with gil:
                    #    print 'Audio need resample, but unable to open it.'
                    avcodec_free_frame(&frame)
                    return -1

                vs.resample_sample_fmt = frame.format
                vs.resample_channel_layout = frame.channel_layout
                vs.resample_sample_rate = frame.sample_rate

            if audio_resample:
                osize = av_get_bytes_per_sample(<AVSampleFormat>mix_sample_fmt)
                nb_samples = frame.nb_samples
                out_size = av_samples_get_buffer_size(&out_linesize,
                        mix_channels, nb_samples,
                        <AVSampleFormat>mix_sample_fmt, 0)
                tmp_out = av_realloc(vs.audio_buf, out_size)
                if tmp_out == NULL:
                    #with gil:
                    #    print 'Unable to realloc audio buffer'
                    avcodec_free_frame(&frame)
                    return -1
                vs.audio_buf = <uint8_t *>tmp_out

                out_samples = avresample_convert(vs.avr,
                        &vs.audio_buf,
                        out_linesize, nb_samples,
                        frame.data,
                        frame.linesize[0],
                        frame.nb_samples)

                if out_samples < 0:
                    #with gil:
                    #    print 'avresample_convert() failed'
                    return -1

                data_size = out_samples * osize * mix_channels

            else:
                vs.audio_buf = frame.data[0]


            pts = vs.audio_clock
            memcpy(pts_ptr, &pts, sizeof(double))
            n = mix_channels * av_get_bytes_per_sample(<AVSampleFormat>mix_sample_fmt)
            vs.audio_clock += <double>data_size / <double>(n * mix_rate)

            # We have data, return it and come back for more later */
            avcodec_free_frame(&frame)
            return data_size

        if pkt.data != flush_pkt.data:
            av_free_packet(pkt)

        if vs.quit:
            avcodec_free_frame(&frame)
            return -1

        if packet_queue_get(&vs.audioq, pkt, 0) < 0:
            avcodec_free_frame(&frame)
            return -1

        if pkt.data == flush_pkt.data:
            avcodec_flush_buffers(vs.audio_st.codec)
            continue

        vs.audio_pkt_data = pkt.data
        vs.audio_pkt_size = pkt.size

        if pkt.pts != AV_NOPTS_VALUE:
            vs.audio_clock = av_q2d(vs.audio_st.time_base) * pkt.pts

    avcodec_free_frame(&frame)
    return 0

@cython.cdivision(True)
cdef void audio_callback(int chan, void *stream, int l, void *userdata) nogil:

    cdef VideoState *vs = <VideoState *>userdata
    cdef int len1, audio_size, out_size, isize, osize, isample
    cdef double pts = 0, s

    if vs.quit == 1:
        return

    while l > 0:

        if vs.audio_buf_index >= vs.audio_buf_size:
            # We have already sent all our data; get more
            audio_size = audio_decode_frame(vs, &pts)
            if vs.quit == 1:
                return
            if audio_size < 0:
                # If error, output silence
                vs.audio_buf = vs.silence_buf
                vs.audio_buf_size = sizeof(vs.silence_buf)
            else:
                audio_size = synchronize_audio(vs, <int16_t*>vs.audio_buf,
                        audio_size, pts)
                vs.audio_buf_size = audio_size
            vs.audio_buf_index = 0

        len1 = vs.audio_buf_size - vs.audio_buf_index
        if len1 > l:
            len1 = l
        memcpy(stream, <uint8_t *>vs.audio_buf + vs.audio_buf_index, len1)
        l -= len1
        stream += len1
        vs.audio_buf_index += len1

    if vs.quit == 1:
        return


cdef void refresh_timer_cb(void *data) nogil:
    cdef VideoState *vs = <VideoState *>data
    event_queue_put_fast(&vs.eq, FF_REFRESH_EVENT, vs)

cdef void schedule_refresh(VideoState *vs, int delay) nogil:
    cdef Event *e = event_create()
    e.name = FF_SCHEDULE_EVENT
    e.userdata = vs
    e.callback = <event_callback_t>refresh_timer_cb
    e.delay = delay
    event_queue_put(&vs.eq, e)

cdef void video_refresh_timer(void *userdata) nogil:

    cdef VideoState *vs = <VideoState *>userdata
    cdef VideoPicture *vp
    cdef double actual_delay, delay, sync_threshold, ref_clock, diff


    if vs.video_st:
        if vs.pictq_size == 0:
            schedule_refresh(vs, 1)
        else:
            vp = &vs.pictq[vs.pictq_rindex]

            vs.video_current_pts = vp.pts
            vs.video_current_pts_time = av_gettime()

            delay = vp.pts - vs.frame_last_pts # the pts from last time
            if delay <= 0 or delay >= 1.0:
                # if incorrect delay, use previous one
                delay = vs.frame_last_delay

            # save for next time
            vs.frame_last_delay = delay
            vs.frame_last_pts = vp.pts

            # update delay to sync to audio if not master source
            if vs.av_sync_type != AV_SYNC_VIDEO_MASTER:
                ref_clock = get_master_clock(vs)
                diff = vp.pts - ref_clock

            # Skip or repeat the frame. Take delay into account
            # FFPlay still doesn't "know if this vs the best guess."
            sync_threshold = delay if (delay > AV_SYNC_THRESHOLD) else AV_SYNC_THRESHOLD
            if fabs(diff) < AV_NOSYNC_THRESHOLD:
                if diff <= -sync_threshold:
                    delay = 0
                elif diff >= sync_threshold:
                    delay = 2 * delay

            vs.frame_timer += delay
            # computer the REAL delay
            actual_delay = vs.frame_timer - (av_gettime() / 1000000.0)
            if actual_delay < 0.010:
                # Really it should skip the picture instead
                actual_delay = 0.010

            schedule_refresh(vs, <int>(actual_delay * 1000 + 0.5))

            # show the picture!
            #video_display(vs)

            # update queue for next picture!
            vs.pictq_rindex += 1
            if vs.pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE:
                vs.pictq_rindex = 0

            SDL_LockMutex(vs.pictq_mutex)
            vs.pictq_size -= 1
            SDL_CondSignal(vs.pictq_cond)
            SDL_UnlockMutex(vs.pictq_mutex)

    else:
        schedule_refresh(vs, 100)

cdef void alloc_picture(void *userdata) nogil:
    cdef VideoState *vs = <VideoState *>userdata
    cdef VideoPicture *vp

    vp = &vs.pictq[vs.pictq_windex]
    if vp.bmp:
        av_freep(&vp.ff_data)
        avcodec_free_frame(&vp.bmp)
        return
    vp.width = vs.video_st.codec.width
    vp.height = vs.video_st.codec.height

    vp.ff_data_size = avpicture_get_size(AV_PIX_FMT_RGB24, vp.width, vp.height)
    vp.ff_data = <unsigned char *>av_malloc(vp.ff_data_size * sizeof(unsigned char))
    vp.bmp = avcodec_alloc_frame()
    avpicture_fill(<AVPicture *>vp.bmp, vp.ff_data, AV_PIX_FMT_RGB24,
            vp.width, vp.height)

    SDL_LockMutex(vs.pictq_mutex)
    vp.allocated = 1
    SDL_CondSignal(vs.pictq_cond)
    SDL_UnlockMutex(vs.pictq_mutex)

cdef int queue_picture(VideoState *vs, AVFrame *pFrame, double pts) nogil:
    cdef VideoPicture *vp
    cdef int dst_pix_fmt
    cdef AVPicture pict
    cdef SDL_UserEvent event

    # wait until we have space for a new pic
    SDL_LockMutex(vs.pictq_mutex)
    while vs.pictq_size >= VIDEO_PICTURE_QUEUE_SIZE and not vs.quit:
        SDL_CondWait(vs.pictq_cond, vs.pictq_mutex)
    SDL_UnlockMutex(vs.pictq_mutex)

    if vs.quit:
        return -1

    # windex vs set to 0 initially
    vp = &vs.pictq[vs.pictq_windex]

    # allocate or resize the buffer!
    if vp.bmp == NULL or \
         vp.width != vs.video_st.codec.width or \
         vp.height != vs.video_st.codec.height:
        vp.allocated = 0

        # we have to do it in the main thread
        event_queue_put_fast(&vs.eq, FF_ALLOC_EVENT, vs)

        # wait until we have a picture allocated
        SDL_LockMutex(vs.pictq_mutex)
        while not vp.allocated and not vs.quit:
            SDL_CondWait(vs.pictq_cond, vs.pictq_mutex)
        SDL_UnlockMutex(vs.pictq_mutex)

        if vs.quit:
            return -1


    # We have a place to put our picture on the queue
    # If we are skipping a frame, do we set this to null
    # but still return vp.allocated = 1?

    cdef int w, h

    if vp.bmp != NULL:

        dst_pix_fmt = AV_PIX_FMT_RGB24

        # Convert the image into YUV format that SDL uses
        if vs.img_convert_ctx == NULL:
            w = vs.video_st.codec.width
            h = vs.video_st.codec.height
            vs.img_convert_ctx = sws_getContext(w, h,
                    vs.video_st.codec.pix_fmt, w, h,
                    dst_pix_fmt, 4, NULL, NULL, NULL)
            if vs.img_convert_ctx == NULL:
                #with gil:
                #    print 'Cannot initialize the conversion context!'
                return -1

        sws_scale(vs.img_convert_ctx,
                <const_uint8_ptr>pFrame.data, pFrame.linesize,
                0, vs.video_st.codec.height, vp.bmp.data, vp.bmp.linesize)

        vp.pts = pts

        # now we inform our display thread that we have a pic ready
        vs.pictq_windex += 1
        if vs.pictq_windex == VIDEO_PICTURE_QUEUE_SIZE:
            vs.pictq_windex = 0

        SDL_LockMutex(vs.pictq_mutex)
        vs.pictq_size += 1
        SDL_UnlockMutex(vs.pictq_mutex)

    return 0


cdef double synchronize_video(VideoState *vs, AVFrame *src_frame, double pts) nogil:
    cdef double frame_delay
    if pts != 0:
        # if we have pts, set video clock to it
        vs.video_clock = pts
    else:
        # if we aren't given a pts, set it to the clock
        pts = vs.video_clock
    # update the video clock */
    frame_delay = av_q2d(vs.video_st.codec.time_base)
    # if we are repeating a frame, adjust clock accordingly */
    frame_delay += src_frame.repeat_pict * (frame_delay * 0.5)
    vs.video_clock += frame_delay
    return pts


cdef int our_get_buffer(AVCodecContext *c, AVFrame *pic) nogil:
    cdef int ret = avcodec_default_get_buffer(c, pic)
    cdef uint64_t *pts = <uint64_t*>av_malloc(sizeof(uint64_t))
    memcpy(pts, &global_video_pkt_pts, sizeof(uint64_t))
    pic.opaque = pts
    return ret


cdef void our_release_buffer(AVCodecContext *c, AVFrame *pic) nogil:
    if pic != NULL:
        av_freep(&pic.opaque)
    avcodec_default_release_buffer(c, pic)


cdef int video_thread(void *arg) nogil:
    cdef VideoState *vs = <VideoState *>arg
    cdef AVPacket pkt1, *packet = &pkt1
    cdef int len1, frameFinished = 0
    cdef AVFrame *pFrame
    cdef double pts, ptst = 0

    pFrame = avcodec_alloc_frame()

    while True:
        if packet_queue_get(&vs.videoq, packet, 1) < 0:
            # means we quit getting packets
            break

        if packet.data == flush_pkt.data:
            avcodec_flush_buffers(vs.video_st.codec)
            continue

        pts = 0

        # Save global pts to be stored in pFrame
        global_video_pkt_pts = packet.pts
        # Decode video frame
        len1 = avcodec_decode_video2(
                vs.video_st.codec, pFrame, &frameFinished, packet)
        if packet.dts == AV_NOPTS_VALUE and pFrame.opaque:
            memcpy(&ptst, pFrame.opaque, sizeof(uint64_t))
            if ptst != AV_NOPTS_VALUE:
                pts = ptst
        elif packet.dts != AV_NOPTS_VALUE:
            pts = packet.dts
        else:
            pts = 0

        pts *= av_q2d(vs.video_st.time_base)

        av_free_packet(packet)

        # Did we get a video frame?
        if frameFinished:
            pts = synchronize_video(vs, pFrame, pts)
            if queue_picture(vs, pFrame, pts) < 0:
                break

    avcodec_free_frame(&pFrame)
    return 0

cdef void stream_component_close(VideoState *vs, int stream_index) with gil:
    cdef AVCodecContext *codecCtx
    codecCtx = vs.pFormatCtx.streams[stream_index].codec
    avcodec_close(codecCtx)

@cython.cdivision(True)
cdef int stream_component_open(VideoState *vs, int stream_index) with gil:
    cdef AVFormatContext *pFormatCtx = vs.pFormatCtx
    cdef AVCodecContext *codecCtx
    cdef AVCodec *codec
    cdef int ret
    #cdef SDL_AudioSpec wanted_spec, spec

    if stream_index < 0 or stream_index >= pFormatCtx.nb_streams:
        return -1

    # Get a pointer to the codec context for the video stream
    codecCtx = pFormatCtx.streams[stream_index].codec
    cdef bytes filename

    if codecCtx.codec_type == AVMEDIA_TYPE_AUDIO:
        # Attach effect to that chunk
        # search an empty channel
        vs.audio_channel = -1
        for i in xrange(MIX_CHANNELS_MAX):
            if mix_channels_usage[i] == 0:
                mix_channels_usage[i] = 1
                vs.audio_channel = i
                break

        if vs.audio_channel == -1:
            print 'No more audio channel available'
            return -1

        # Create Mix Chunk from that entry
        filename = <bytes>join(dirname(__file__), 'silence.wav')
        vs.audio_chunk = Mix_LoadWAV_RW(
                SDL_RWFromMem(<char *><bytes>silence_wav, len(silence_wav)), 1)
        if vs.audio_chunk == NULL:
            print 'Unable to load chunk'
            return -1

        with nogil:
            SDL_LockAudio()
            ret = Mix_RegisterEffect(vs.audio_channel, audio_callback, NULL, vs)
            SDL_UnlockAudio()
        if ret < 0:
            print 'Unable to register effect!'
            return -1

        # Set audio settings from codec info
        '''
        wanted_spec.freq = codecCtx.sample_rate
        wanted_spec.format = AUDIO_S16SYS
        wanted_spec.channels = codecCtx.channels
        wanted_spec.silence = 0
        wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE
        wanted_spec.callback = audio_callback
        wanted_spec.userdata = vs
        '''

        vs.audio_hw_buf_size = SDL_AUDIO_BUFFER_SIZE

    codec = avcodec_find_decoder(codecCtx.codec_id)
    if codec == NULL or avcodec_open2(codecCtx, codec, NULL) < 0:
        print 'Unsupported codec!'
        return -1

    if codecCtx.codec_type == AVMEDIA_TYPE_AUDIO:
        vs.audioStream = stream_index
        vs.audio_st = pFormatCtx.streams[stream_index]
        vs.audio_buf_size = 0
        vs.audio_buf_index = 0

        # averaging filter for audio sync
        vs.audio_diff_avg_coef = exp(log(0.01 / AUDIO_DIFF_AVG_NB))
        vs.audio_diff_avg_count = 0
        # Correct audio only if larger error than this
        vs.audio_diff_threshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / codecCtx.sample_rate

        memset(&vs.audio_pkt, 0, sizeof(vs.audio_pkt))
        packet_queue_init(&vs.audioq)

        with nogil:
            Mix_PlayChannel(vs.audio_channel, vs.audio_chunk, -1)

    elif codecCtx.codec_type == AVMEDIA_TYPE_VIDEO:
        vs.videoStream = stream_index
        vs.video_st = pFormatCtx.streams[stream_index]

        vs.frame_timer = <double>av_gettime() / 1000000.0
        vs.frame_last_delay = 40e-3
        vs.video_current_pts_time = av_gettime()

        packet_queue_init(&vs.videoq)
        vs.video_tid = SDL_CreateThread(video_thread, 'video', vs)
        codecCtx.get_buffer = our_get_buffer
        codecCtx.release_buffer = our_release_buffer

    else:
        pass

cdef int decode_interrupt_cb(void *not_used) nogil:
    if global_video_state != NULL:
        return global_video_state.quit
    return 0

cdef int decode_thread(void *arg) nogil:
    cdef VideoState *vs = <VideoState *>arg
    cdef AVFormatContext *pFormatCtx = NULL
    cdef AVPacket pkt1, *packet = &pkt1
    cdef int video_index = -1
    cdef int audio_index = -1
    cdef int i, codec_type
    cdef int stream_index = -1
    cdef int64_t seek_target = 0
    cdef AVRational AV_TIME_BASE_Q
    cdef int seek_flags = 0

    AV_TIME_BASE_Q.num = 1
    AV_TIME_BASE_Q.den = 1000000

    vs.videoStream = -1
    vs.audioStream = -1

    global_video_state = vs

    # Open video file
    # will interrupt blocking functions if we quit!
    pFormatCtx = avformat_alloc_context()
    pFormatCtx.interrupt_callback.callback = decode_interrupt_cb
    pFormatCtx.interrupt_callback.opaque = NULL
    if avformat_open_input(&pFormatCtx, vs.filename, NULL, NULL) != 0:
        avformat_free_context(pFormatCtx)
        return -1 # Couldn't open file

    vs.pFormatCtx = pFormatCtx

    # Retrieve stream information
    if avformat_find_stream_info(pFormatCtx, NULL) < 0:
        return -1 # Couldn't find stream information

    # Dump information about file onto standard error
    av_dump_format(pFormatCtx, 0, vs.filename, 0)

    # Find the first video stream
    for i in xrange(pFormatCtx.nb_streams):
        codec_type = pFormatCtx.streams[i].codec.codec_type
        if codec_type == AVMEDIA_TYPE_VIDEO and video_index < 0:
            video_index = i

        if codec_type == AVMEDIA_TYPE_AUDIO and audio_index < 0:
            audio_index = i

    with gil:
        if audio_index >= 0:
            stream_component_open(vs, audio_index)

        if video_index >= 0:
            stream_component_open(vs, video_index)

        if vs.videoStream < 0 or vs.audioStream < 0:
            if vs.audioStream < 0 and audio_index >= 0:
                print '%s: could not open codecs for audio' % vs.filename
            if vs.videoStream < 0 and video_index >= 0:
                print '%s: could not open codecs for video' % vs.filename
            if vs.videoStream < 0:
                print 'LEAVE EVERYTHING!'
                event_queue_put_fast(&vs.eq, FF_QUIT_EVENT, vs)
                return 0


    # main decode loop
    while True:

        if vs.quit:
            break

        # seek stuff goes here
        if vs.seek_req:
            stream_index = -1

            if vs.videoStream >= 0:
                stream_index = vs.videoStream
            elif vs.audioStream >= 0:
                stream_index = vs.audioStream

            if stream_index >= 0:

                seek_target = <int64_t>(<double>pFormatCtx.duration * vs.seek_percent)

                seek_target = av_rescale_q(
                        seek_target, AV_TIME_BASE_Q,
                        pFormatCtx.streams[stream_index].time_base)

                # FIXME, that doesnt work yet
                if seek_target > get_master_clock(vs) * AV_TIME_BASE:
                    seek_flags = 0
                else:
                    seek_flags = AVSEEK_FLAG_BACKWARD

            if av_seek_frame(vs.pFormatCtx, stream_index,
                    seek_target, seek_flags) < 0:
                pass
            else:
                if vs.audioStream >= 0:
                    packet_queue_flush(&vs.audioq)
                    packet_queue_put(&vs.audioq, &flush_pkt)

                if vs.videoStream >= 0:
                    packet_queue_flush(&vs.videoq)
                    packet_queue_put(&vs.videoq, &flush_pkt)

            vs.seek_req = 0

        if vs.audioq.size > MAX_AUDIOQ_SIZE or vs.videoq.size > MAX_VIDEOQ_SIZE:
            SDL_Delay(10)
            continue

        if av_read_frame(vs.pFormatCtx, packet) < 0:
            break

        # Is this a packet from the video stream?
        if packet.stream_index == vs.videoStream:
            packet_queue_put(&vs.videoq, packet)
        elif packet.stream_index == vs.audioStream:
            packet_queue_put(&vs.audioq, packet)

    # all done - wait for it
    cdef int canquit = 0
    while vs.quit == 0:
        canquit = 0
        SDL_LockMutex(vs.audioq.mutex)
        if vs.audioq.nb_packets == 0:
            canquit += 1
        SDL_UnlockMutex(vs.audioq.mutex)
        SDL_LockMutex(vs.videoq.mutex)
        if vs.videoq.nb_packets == 0:
            canquit += 1
        SDL_UnlockMutex(vs.videoq.mutex)
        if canquit == 2:
            break
        SDL_Delay(100)

    event_queue_put_fast(&vs.eq, FF_QUIT_EVENT, vs)

    # video finished, set everythign to quit.
    vs.quit = 1
    vs.audioq.quit = 1
    vs.videoq.quit = 1

    # unlock queue and video in case of.
    SDL_LockMutex(vs.videoq.mutex)
    SDL_CondSignal(vs.videoq.cond)
    SDL_UnlockMutex(vs.videoq.mutex)
    SDL_LockMutex(vs.pictq_mutex)
    SDL_CondSignal(vs.pictq_cond)
    SDL_UnlockMutex(vs.pictq_mutex)

    if vs.video_tid != NULL:
        SDL_WaitThread(vs.video_tid, NULL)
        vs.video_tid = NULL

    return 0

cdef class ScheduledEvent:
    cdef Event *event

class FFVideoException(Exception):
    pass

cdef class FFVideo:
    cdef bytes filename
    cdef VideoState *vs
    cdef list events

    def __cinit__(self, filename):
        self.filename = None
        self.vs = NULL
        self.events = []

    def __init__(self, filename):
        self.filename = filename

    property is_open:
        def __get__(self):
            return self.vs != NULL

    def open(self):
        cdef int i
        cdef VideoState *vs

        ffmpeg_ensure_init()

        # allocate memory for video state
        self.vs = vs = <VideoState *>av_mallocz(sizeof(VideoState));
        if vs == NULL:
            raise FFVideoException('Unable to allocate memory (1)')

        # initialize video state
        event_queue_init(&vs.eq)
        memcpy(vs.filename, <char *>self.filename, imin(sizeof(vs.filename), len(self.filename)))
        vs.pictq_mutex = SDL_CreateMutex()
        vs.pictq_cond = SDL_CreateCond()
        vs.audio_channel = -1

        schedule_refresh(vs, 40)

        vs.av_sync_type = DEFAULT_AV_SYNC_TYPE
        with nogil:
            vs.parse_tid = SDL_CreateThread(decode_thread, 'audio', vs)
        if vs.parse_tid == NULL:
            av_free(vs)
            self.vs = NULL

        av_init_packet(&flush_pkt)
        flush_pkt.data = <uint8_t *><char *>'FLUSH'

    cdef void free(self):
        cdef VideoState *vs = self.vs
        if vs == NULL:
            return

        # ensure that nobody will wait on a queue get
        vs.quit = 1
        vs.audioq.quit = 1
        vs.videoq.quit = 1
        if vs.audio_channel != -1:
            Mix_HaltChannel(vs.audio_channel)
            mix_channels_usage[vs.audio_channel] = 0
            with nogil:
                SDL_LockAudio()
                Mix_UnregisterAllEffects(vs.audio_channel)
                SDL_UnlockAudio()

        if vs.parse_tid != NULL:
            with nogil:
                SDL_WaitThread(vs.parse_tid, NULL)
            vs.parse_tid = NULL

        event_queue_clean(&vs.eq)
        packet_queue_clean(&vs.audioq)
        packet_queue_clean(&vs.videoq)
        if vs.pictq_mutex != NULL:
            SDL_DestroyMutex(vs.pictq_mutex)
        if vs.pictq_cond != NULL:
            SDL_DestroyCond(vs.pictq_cond)

        if vs.audioStream >= 0:
            stream_component_close(vs, vs.audioStream)
            av_free_packet(&vs.audio_pkt)
            vs.audioStream = -1
        if vs.videoStream >= 0:
            stream_component_close(vs, vs.videoStream)
            vs.videoStream = -1
        if vs.avr != NULL:
            avresample_free(&vs.avr)
        if vs.audio_buf != NULL:
            if vs.audio_buf != vs.silence_buf:
                av_freep(&vs.audio_buf)
        if vs.pFormatCtx != NULL:
            vs.video_st = NULL
            vs.audio_st = NULL
            avformat_close_input(&vs.pFormatCtx)
            #avformat_free_context(vs.pFormatCtx)
            vs.pFormatCtx = NULL
        if vs.img_convert_ctx != NULL:
            sws_freeContext(vs.img_convert_ctx)
            vs.img_convert_ctx = NULL

        cdef VideoPicture *vp
        cdef int index
        for index in xrange(VIDEO_PICTURE_QUEUE_SIZE):
            vp = &vs.pictq[index]
            if vp.ff_data != NULL:
                av_freep(&vp.ff_data)
            if vp.bmp != NULL:
                avcodec_free_frame(&vp.bmp)

        av_freep(&vs)
        self.vs = NULL

        # flush events
        cdef ScheduledEvent se
        for item in self.events:
            itime, se = item
            free(se.event)
        del self.events[:]

    cdef void update(self):
        cdef Event *event
        cdef ScheduledEvent se
        cdef int64_t curtime, itime

        if self.vs == NULL:
            return

        curtime = av_gettime()
        # check our own events
        for item in self.events[:]:
            itime, se = item
            if curtime < itime:
                continue
            self.events.remove(item)
            se.event.callback(se.event.userdata)
            free(se.event)

        # read thread event
        while True:
            event = event_queue_get(&self.vs.eq)
            if event == NULL:
                return
            if event.name == FF_ALLOC_EVENT:
                alloc_picture(event.userdata)
            elif event.name == FF_REFRESH_EVENT:
                if self.vs.quit == 0:
                    video_refresh_timer(event.userdata)
            elif event.name == FF_QUIT_EVENT:
                self.vs.quit = 1
                self.free()
                free(event)
                break
            elif event.name == FF_SCHEDULE_EVENT:
                se = ScheduledEvent()
                se.event = event
                self.events.append((curtime + event.delay * 1000., se))
                continue # don't free event in that case
            free(event)

    cpdef int get_width(self):
        cdef VideoState *vs = self.vs
        if vs == NULL:
            return -1
        if vs.video_st == NULL or vs.video_st.codec == NULL:
            return -1
        return vs.video_st.codec.width

    cpdef int get_height(self):
        cdef VideoState *vs = self.vs
        if vs == NULL:
            return -1
        if vs.video_st == NULL or vs.video_st.codec == NULL:
            return -1
        return vs.video_st.codec.height

    def get_next_frame(self):
        cdef int size, y, index
        cdef int width, height
        cdef AVFrame *rgb
        cdef VideoState *vs = self.vs
        cdef bytes ret

        if vs == NULL:
            return

        self.update()

        if self.get_width() == -1:
            return

        ret = None
        with nogil:
            SDL_LockMutex(vs.pictq_mutex)

        width = self.get_width()
        height = self.get_height()
        size = width * height * 3
        rgb = vs.pictq[vs.pictq_rindex].bmp
        if rgb:
            ret = (<char *>rgb.data[0])[:size]

        with nogil:
            SDL_UnlockMutex(vs.pictq_mutex)

        return ret

    def stop(self):
        self.free()

    def get_duration(self):
        if self.vs == NULL or self.vs.pFormatCtx == NULL:
            return 0
        return self.vs.pFormatCtx.duration / AV_TIME_BASE

    def get_position(self):
        if self.vs == NULL:
            return 0
        return get_master_clock(self.vs)

    def seek(self, percent):
        if self.vs == NULL:
            return
        self.vs.seek_percent = percent
        self.vs.seek_req = 1

    def set_volume(self, int volume):
        if self.vs == NULL or self.vs.audio_channel == -1:
            return
        if volume < 0:
            volume = 0
        if volume > 128:
            volume = 128
        with nogil:
            SDL_LockAudio()
            Mix_Volume(self.vs.audio_channel, volume)
            SDL_UnlockAudio()

    def get_volume(self):
        cdef int volume = 0
        if self.vs == NULL or self.vs.audio_channel == -1:
            return
        with nogil:
            SDL_LockAudio()
            volume = Mix_Volume(self.vs.audio_channel, -1)
            SDL_UnlockAudio()
        return volume

