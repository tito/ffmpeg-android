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

'''

include '_ffmpeg.pxi'

from time import time
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
    AVFormatContext *pFormatCtx
    int             videoStream
    int             audioStream

    int             av_sync_type
    double          external_clock # external clock base
    int64_t         external_clock_time
    int             seek_req
    int             seek_flags
    int64_t         seek_pos
    double          audio_clock
    AVStream        *audio_st
    PacketQueue     audioq
    uint8_t         audio_buf[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]
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
    unsigned char   *pixels
    Mix_Chunk       *audio_chunk
    int             audio_channel


cdef VideoState *global_video_state = NULL

#
# Mixer
# We need a way to mix several raw channels together.
# The only way was to play an empty sound on each channel, but replace the
# stream by the sound as a sound effect.
#


DEF MIX_CHANNELS_MAX = 32
cdef int mix_audio_have_init = 0
cdef int mix_channels[MIX_CHANNELS_MAX]

cdef int mix_audio_init():
    global g_have_register_audio
    if g_have_register_audio == 1:
        return 0
    g_have_register_audio = 1

    if SDL_Init(SDL_INIT_AUDIO) < 0:
        print 'SDL_Init: %s' % SDL_GetError()
        return -1

    if Mix_OpenAudio(44100, AUDIO_S16SYS, 2, 1024):
        print 'Mix_OpenAudio: %s' % SDL_GetError()
        return -1

    memset(mix_channels, 0, sizeof(int) * MIX_CHANNELS_MAX)

    Mix_AllocateChannels(MIX_CHANNELS_MAX)

    return 0

#
# User event queue, to communicate between thread and python class
# No python used, to be able at some time to remove GIL usage.
#

cdef void event_queue_init(EventQueue *q):
    memset(q, 0, sizeof(EventQueue))
    q.mutex = SDL_CreateMutex()

cdef Event *event_create():
    cdef Event *event = <Event *>malloc(sizeof(Event))
    memset(event, 0, sizeof(Event))
    return event

cdef void event_queue_put(EventQueue *q, Event *e):
    with nogil:
        SDL_LockMutex(q.mutex)
    if q.last != NULL:
        q.last.next = e
    q.last = e
    if q.first == NULL:
        q.first = e
    with nogil:
        SDL_UnlockMutex(q.mutex)

cdef void event_queue_put_fast(EventQueue *q, int name, void *userdata):
    cdef Event *e = event_create()
    e.name = name
    e.userdata = userdata
    event_queue_put(q, e)

cdef Event *event_queue_get(EventQueue *q):
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

#
# Packet Queue
#

cdef void packet_queue_init(PacketQueue *q):
    memset(q, 0, sizeof(PacketQueue))
    q.mutex = SDL_CreateMutex()
    q.cond = SDL_CreateCond()


cdef int packet_queue_put(PacketQueue *q, AVPacket *pkt):
    cdef AVPacketList *pkt1

    if av_dup_packet(pkt) < 0:
        return -1

    pkt1 = <AVPacketList *>av_malloc(sizeof(AVPacketList))
    if pkt1 == NULL:
        return - 1

    memcpy(&pkt1.pkt, pkt, sizeof(AVPacket))
    pkt1.next = NULL

    with nogil:
        SDL_LockMutex(q.mutex)

    if q.last_pkt == NULL:
        q.first_pkt = pkt1
    else:
        q.last_pkt.next = pkt1
    q.last_pkt = pkt1
    q.nb_packets += 1
    q.size += pkt1.pkt.size

    with nogil:
        SDL_CondSignal(q.cond) 
        SDL_UnlockMutex(q.mutex)

    return 0


cdef int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block):
    cdef AVPacketList *pkt1
    cdef int ret = -1

    with nogil:
        SDL_LockMutex(q.mutex)

    while True:

        # FIXME!!
        #if ctx.quit:
        #    ret = -1
        #    break

        pkt1 = q.first_pkt
        if pkt1 != NULL:
            q.first_pkt = pkt1.next
            if q.first_pkt == NULL:
                q.last_pkt = NULL
            q.nb_packets -= 1
            q.size -= pkt1.pkt.size
            memcpy(pkt, &pkt1.pkt, sizeof(AVPacket))
            av_free(pkt1)
            ret = 1
            break
        elif block == 0:
            ret = 0
            break
        else:
            with nogil: SDL_CondWait(q.cond, q.mutex)

    with nogil:
        SDL_UnlockMutex(q.mutex)

    return ret


cdef void packet_queue_flush(PacketQueue *q):
    cdef AVPacketList *pkt, *pkt1

    with nogil:
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
    with nogil:
        SDL_UnlockMutex(q.mutex)


cdef double get_audio_clock(VideoState *vs):
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


cdef double get_video_clock(VideoState *vs):
    cdef double delta
    delta = (av_gettime() - vs.video_current_pts_time) / 1000000.0
    return vs.video_current_pts + delta


cdef double get_external_clock(VideoState *vs):
    return av_gettime() / 1000000.0


cdef double get_master_clock(VideoState *vs):
    if vs.av_sync_type == AV_SYNC_VIDEO_MASTER:
        return get_video_clock(vs)
    elif vs.av_sync_type == AV_SYNC_AUDIO_MASTER:
        return get_audio_clock(vs)
    else:
        return get_external_clock(vs)
    

cdef int synchronize_audio(VideoState *vs, short *samples,
		            int samples_size, double pts):
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


cdef int audio_decode_frame(VideoState *vs, uint8_t *audio_buf, int buf_size,
        double *pts_ptr):
    cdef AVPacket *pkt = &vs.audio_pkt
    cdef int len1, data_size, n
    cdef double pts

    while True:

        while vs.audio_pkt_size > 0:

            data_size = buf_size

            len1 = avcodec_decode_audio2(
                    vs.audio_st.codec, <int16_t *>vs.audio_buf, &data_size, 
                    vs.audio_pkt_data, vs.audio_pkt_size)

            if len1 < 0:
                # if error, skip frame
                vs.audio_pkt_size = 0
                break
            vs.audio_pkt_data += len1
            vs.audio_pkt_size -= len1
            if data_size <= 0:
                # No data yet, get more frames
                continue
            pts = vs.audio_clock
            memcpy(pts_ptr, &pts, sizeof(double))
            n = 2 * vs.audio_st.codec.channels
            vs.audio_clock += <double>data_size / <double>(n *
                    vs.audio_st.codec.sample_rate)

            # We have data, return it and come back for more later */
            return data_size

        if pkt.data:
            av_free_packet(pkt)

        # FIXME
        # if ctx.quit:
        #    return -1

        if packet_queue_get(&vs.audioq, pkt, 1) < 0:
            return -1

        if pkt.data == flush_pkt.data:
            avcodec_flush_buffers(vs.audio_st.codec)

        vs.audio_pkt_data = pkt.data
        vs.audio_pkt_size = pkt.size

        if pkt.pts != AV_NOPTS_VALUE:
            vs.audio_clock = av_q2d(vs.audio_st.time_base) * pkt.pts

    return 0


cdef void audio_callback(int chan, void *stream, int l, void *userdata) with gil:

    cdef VideoState *vs = <VideoState *>userdata
    cdef int len1, audio_size
    cdef double pts = 0

    while l > 0:

        if vs.audio_buf_index >= vs.audio_buf_size:
            # We have already sent all our data; get more
            audio_size = audio_decode_frame(vs, vs.audio_buf,
                    sizeof(vs.audio_buf), &pts)
            if audio_size < 0:
                # If error, output silence
                vs.audio_buf_size = 1024
                memset(vs.audio_buf, 0, vs.audio_buf_size)
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

cdef void refresh_timer_cb(void *data):
    cdef VideoState *vs = <VideoState *>data
    print 'refresh_timer_cb()'
    event_queue_put_fast(&vs.eq, FF_REFRESH_EVENT, vs)

cdef void schedule_refresh(VideoState *vs, int delay):
    cdef Event *e = event_create()
    e.name = FF_SCHEDULE_EVENT
    e.userdata = vs
    e.callback = <event_callback_t>refresh_timer_cb
    e.delay = delay
    event_queue_put(&vs.eq, e)
    print 'schedule_refresh()'

cdef void video_display(VideoState *vs):
    cdef int width = vs.video_st.codec.width
    cdef int height = vs.video_st.codec.height
    cdef int index, y
    cdef AVFrame *rgb

    if width == -1:
        return

    print 'video_display()', vs.pictq_rindex

    with nogil:
        SDL_LockMutex(vs.pictq_mutex)

    size = width * height * 3
    if vs.pixels == NULL:
        vs.pixels = <unsigned char *>malloc(size * sizeof(unsigned char))
    if vs.pixels != NULL:
        rgb = vs.pictq[vs.pictq_rindex].bmp
        if rgb != NULL:
            # copy frame into pixels
            index = 0
            for y in xrange(height):
                memcpy(&vs.pixels[index], rgb.data[0] + \
                        y * rgb.linesize[0], width * 3)
                index += width * 3

    with nogil:
        SDL_UnlockMutex(vs.pictq_mutex)


cdef void video_refresh_timer(void *userdata):

    cdef VideoState *vs = <VideoState *>userdata
    cdef VideoPicture *vp
    cdef double actual_delay, delay, sync_threshold, ref_clock, diff

    print 'video_refresh_timer()'
    
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
            video_display(vs)
            
            # update queue for next picture!
            vs.pictq_rindex += 1
            if vs.pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE:
                vs.pictq_rindex = 0
            
            with nogil:
                SDL_LockMutex(vs.pictq_mutex)
            vs.pictq_size -= 1
            with nogil:
                SDL_CondSignal(vs.pictq_cond)
                SDL_UnlockMutex(vs.pictq_mutex)
        
    else:
        schedule_refresh(vs, 100)
    
cdef void alloc_picture(void *userdata):
    cdef VideoState *vs = <VideoState *>userdata
    cdef VideoPicture *vp

    vp = &vs.pictq[vs.pictq_windex]
    if vp.bmp:
        assert(0)
        free(vp.bmp)
    vp.width = vs.video_st.codec.width
    vp.height = vs.video_st.codec.height

    vp.ff_data_size = avpicture_get_size(PixelFormats.RGB24,
            vp.width, vp.height)
    vp.ff_data = <unsigned char *>av_malloc(vp.ff_data_size * sizeof(unsigned char))
    vp.bmp = avcodec_alloc_frame()
    avpicture_fill(<AVPicture *>vp.bmp, vp.ff_data, PixelFormats.RGB24,
            vp.width, vp.height)

    print 'alloc_picture() lock'
    with nogil:
        SDL_LockMutex(vs.pictq_mutex)
    vp.allocated = 1
    print 'alloc_picture() send signal'
    with nogil:
        SDL_CondSignal(vs.pictq_cond)
        SDL_UnlockMutex(vs.pictq_mutex)
    print 'alloc_picture() sent & unlock'

cdef int queue_picture(VideoState *vs, AVFrame *pFrame, double pts):
    cdef VideoPicture *vp
    cdef int dst_pix_fmt
    cdef AVPicture pict
    cdef SDL_UserEvent event

    print 'queue_picture()', vs.pictq_size

    # wait until we have space for a new pic
    with nogil:
        SDL_LockMutex(vs.pictq_mutex)
    print 'queue_picture() after lock'
    while vs.pictq_size >= VIDEO_PICTURE_QUEUE_SIZE and not vs.quit:
        print 'queue_picture() inside while', vs.pictq_size, VIDEO_PICTURE_QUEUE_SIZE, vs.quit
        with nogil:
            SDL_CondWait(vs.pictq_cond, vs.pictq_mutex)
    
    with nogil:
        SDL_UnlockMutex(vs.pictq_mutex)

    print 'queue_picture() after mutex'

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
        with nogil:
            SDL_LockMutex(vs.pictq_mutex)
        print 'here !!!'
        while not vp.allocated and not vs.quit:
            with nogil:
                SDL_CondWait(vs.pictq_cond, vs.pictq_mutex)
        
        SDL_UnlockMutex(vs.pictq_mutex)
        if vs.quit:
            return -1
        
    
    # We have a place to put our picture on the queue
    # If we are skipping a frame, do we set this to null 
    # but still return vp.allocated = 1?

    cdef int w, h

    if vp.bmp != NULL:

        dst_pix_fmt = PixelFormats.RGB24

        # Convert the image into YUV format that SDL uses
        if vs.img_convert_ctx == NULL:
            w = vs.video_st.codec.width
            h = vs.video_st.codec.height
            vs.img_convert_ctx = sws_getContext(w, h, 
                    vs.video_st.codec.pix_fmt, w, h, 
                    dst_pix_fmt, 4, NULL, NULL, NULL)
            if vs.img_convert_ctx == NULL:
                print 'Cannot initialize the conversion context!'
                return -1
        
        sws_scale(vs.img_convert_ctx, pFrame.data, pFrame.linesize,
                    0, vs.video_st.codec.height, vp.bmp.data, vp.bmp.linesize)
        
        vp.pts = pts

        # now we inform our display thread that we have a pic ready
        vs.pictq_windex += 1
        if vs.pictq_windex == VIDEO_PICTURE_QUEUE_SIZE:
            vs.pictq_windex = 0
        
        with nogil:
            SDL_LockMutex(vs.pictq_mutex)
        vs.pictq_size += 1
        SDL_UnlockMutex(vs.pictq_mutex)
    
    return 0
        

cdef double synchronize_video(VideoState *vs, AVFrame *src_frame, double pts):
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


cdef int our_get_buffer(AVCodecContext *c, AVFrame *pic):
    cdef int ret = avcodec_default_get_buffer(c, pic)
    cdef uint64_t *pts = <uint64_t*>av_malloc(sizeof(uint64_t))
    memcpy(pts, &global_video_pkt_pts, sizeof(uint64_t))
    pic.opaque = pts
    return ret


cdef void our_release_buffer(AVCodecContext *c, AVFrame *pic):
    # XXX FIXME!
    #if pic != NULL: av_freep(pic.opaque)
    avcodec_default_release_buffer(c, pic)
        

cdef int video_thread(void *arg) with gil:
    cdef VideoState *vs = <VideoState *>arg
    cdef AVPacket pkt1, *packet = &pkt1
    cdef int len1, frameFinished = 0
    cdef AVFrame *pFrame
    cdef double pts, ptst = 0

    pFrame = avcodec_alloc_frame()

    print 'video_thread() started'

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
        len1 = avcodec_decode_video2(vs.video_st.codec, pFrame, &frameFinished, 
                                packet)
        if packet.dts == AV_NOPTS_VALUE and pFrame.opaque:
            memcpy(&ptst, pFrame.opaque, sizeof(uint64_t))
            if ptst != AV_NOPTS_VALUE:
                pts = ptst
        elif packet.dts != AV_NOPTS_VALUE:
            pts = packet.dts
        else:
            pts = 0
        
        pts *= av_q2d(vs.video_st.time_base)


        # Did we get a video frame?
        if frameFinished:
            print 'video_thread() got frame, synchronize'
            pts = synchronize_video(vs, pFrame, pts)
            print 'video_thread() queue picture'
            if queue_picture(vs, pFrame, pts) < 0:
                break
            
        av_free_packet(packet)
    
    print 'video_thread() leaved'
    av_free(pFrame)
    return 0

cdef int stream_component_open(VideoState *vs, int stream_index):
    cdef AVFormatContext *pFormatCtx = vs.pFormatCtx
    cdef AVCodecContext *codecCtx
    cdef AVCodec *codec
    #cdef SDL_AudioSpec wanted_spec, spec

    if stream_index < 0 or stream_index >= pFormatCtx.nb_streams:
        return -1
    
    # Get a pointer to the codec context for the video stream
    codecCtx = pFormatCtx.streams[stream_index].codec
    cdef bytes filename

    if codecCtx.codec_type == CODEC_TYPE_AUDIO:
        # Init audio
        if mix_audio_init() < 0:
            print 'Unable to init audio'
            return -1

        # Attach effect to that chunk
        # search an empty channel
        vs.audio_channel = -1
        for i in xrange(MIX_CHANNELS_MAX):
            if mix_channels[i] == 0:
                mix_channels[i] = 1
                vs.audio_channel = i
                break

        if vs.audio_channel == -1:
            print 'No more audio channel available'
            return -1

        # Create Mix Chunk from that entry
        filename = <bytes>join(dirname(__file__), 'silence.wav')
        vs.audio_chunk = Mix_LoadWAV(filename)
        if vs.audio_chunk == NULL:
            print 'Unable to load chunk'
            return -1

        if Mix_RegisterEffect(vs.audio_channel, audio_callback, NULL, vs) < 0:
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
    if codec == NULL or avcodec_open(codecCtx, codec) < 0:
        print 'Unsupported codec!'
        return -1
    

    if codecCtx.codec_type == CODEC_TYPE_AUDIO:
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

        Mix_PlayChannel(vs.audio_channel, vs.audio_chunk, -1)

    elif codecCtx.codec_type == CODEC_TYPE_VIDEO:
        vs.videoStream = stream_index
        vs.video_st = pFormatCtx.streams[stream_index]

        vs.frame_timer = <double>av_gettime() / 1000000.0
        vs.frame_last_delay = 40e-3
        vs.video_current_pts_time = av_gettime()

        packet_queue_init(&vs.videoq)
        vs.video_tid = SDL_CreateThread(video_thread, vs)
        codecCtx.get_buffer = our_get_buffer
        codecCtx.release_buffer = our_release_buffer

cdef int decode_interrupt_cb():
    if global_video_state != NULL:
        return global_video_state.quit
    return 0

cdef int decode_thread(void *arg):
    cdef VideoState *vs = <VideoState *>arg
    cdef AVFormatContext *pFormatCtx = NULL
    cdef AVPacket pkt1, *packet = &pkt1
    cdef int video_index = -1
    cdef int audio_index = -1
    cdef int i, codec_type
    cdef int stream_index = -1
    cdef int64_t seek_target = 0
    cdef AVRational AV_TIME_BASE_Q

    AV_TIME_BASE_Q.num = 1
    AV_TIME_BASE_Q.den = 1000000

    vs.videoStream = -1
    vs.audioStream = -1

    global_video_state = vs
    # will interrupt blocking functions if we quit!
    url_set_interrupt_cb(decode_interrupt_cb)

    # Open video file
    if av_open_input_file(&pFormatCtx, vs.filename, NULL, 0, NULL) != 0:
        return -1 # Couldn't open file

    vs.pFormatCtx = pFormatCtx
    
    # Retrieve stream information
    if av_find_stream_info(pFormatCtx) < 0:
        return -1 # Couldn't find stream information
    
    # Dump information about file onto standard error
    dump_format(pFormatCtx, 0, vs.filename, 0)
    
    # Find the first video stream
    for i in xrange(pFormatCtx.nb_streams):
        codec_type = pFormatCtx.streams[i].codec.codec_type
        if codec_type == CODEC_TYPE_VIDEO and video_index < 0:
            video_index = i
        
        if codec_type == CODEC_TYPE_AUDIO and audio_index < 0:
            audio_index = i
        
    if audio_index >= 0:
        stream_component_open(vs, audio_index)
    
    if video_index >= 0:
        stream_component_open(vs, video_index)
         

    if vs.videoStream < 0 or vs.audioStream < 0:
        print '%s: could not open codecs' % vs.filename
        if vs.audioStream < 0:
            print '%s: cannot open for audio' % vs.filename
        if vs.videoStream < 0:
            print '%s: cannot open for video' % vs.filename
        if vs.videoStream < 0:
            event_queue_put_fast(&vs.eq, FF_QUIT_EVENT, vs)
            return 0
    

    # main decode loop
    while True:

        if vs.quit:
            break
        
        # seek stuff goes here
        if vs.seek_req:
            stream_index = -1
            seek_target = vs.seek_pos

            if vs.videoStream >= 0:
                stream_index = vs.videoStream
            elif vs.audioStream >= 0:
                stream_index = vs.audioStream

            if stream_index >= 0:

                seek_target = av_rescale_q(
                        seek_target, AV_TIME_BASE_Q,
                        pFormatCtx.streams[stream_index].time_base)
            
            if not av_seek_frame(vs.pFormatCtx, stream_index,
                    seek_target, vs.seek_flags):
                #print '%s: error while seeking' % vs.pFormatCtx.filename
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
            with nogil: SDL_Delay(10)
            continue
        
        if av_read_frame(vs.pFormatCtx, packet) < 0:
            if pFormatCtx.pb.error == 0:
                with nogil: SDL_Delay(100) # no error wait for user input
                continue
            else:
                break
            
        
        # Is this a packet from the video stream?
        if packet.stream_index == vs.videoStream:
            packet_queue_put(&vs.videoq, packet)
        elif packet.stream_index == vs.audioStream:
            packet_queue_put(&vs.audioq, packet)
        else:
            av_free_packet(packet)
        
    # all done - wait for it
    while vs.quit == 0:
        with nogil: SDL_Delay(100)

    event_queue_put_fast(&vs.eq, FF_QUIT_EVENT, vs)
    
    return 0

cdef int __decode_thread(void *arg) nogil:
    with gil:
        return decode_thread(arg)

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
        PyEval_InitThreads()

    def open(self):
        cdef int i
        cdef VideoState *vs
        global g_have_register

        # ensure that ffmpeg have been registered first
        if g_have_register == 0:
            av_register_all()
            g_have_register = 1

        # allocate memory for video state
        self.vs = vs = <VideoState *>av_mallocz(sizeof(VideoState));
        if vs == NULL:
            raise FFVideoException('Unable to allocate memory (1)')

        # initialize video state
        event_queue_init(&vs.eq)
        memcpy(vs.filename, <char *>self.filename, min(sizeof(vs.filename),
            len(self.filename)))
        vs.pictq_mutex = SDL_CreateMutex()
        vs.pictq_cond = SDL_CreateCond()

        schedule_refresh(vs, 40)

        vs.av_sync_type = DEFAULT_AV_SYNC_TYPE
        with nogil:
            vs.parse_tid = SDL_CreateThread(__decode_thread, vs)
        if vs.parse_tid == NULL:
            av_free(vs)
            self.vs = NULL

        av_init_packet(&flush_pkt)
        flush_pkt.data = <uint8_t *><char *>'FLUSH'

    cdef void update(self):
        cdef Event *event
        cdef int curtime, itime
        cdef ScheduledEvent se

        if self.vs == NULL:
            return

        curtime = av_gettime()
        # check our own events
        for item in self.events[:]:
            itime, se = item
            if curtime < itime:
                continue
            self.events.remove(item)
            #print 'xx execute callback, delay was', se.event.delay
            se.event.callback(se.event.userdata)
            free(se.event)

        # read thread event
        while True:
            event = event_queue_get(&self.vs.eq)
            if event == NULL:
                return
            #print 'execute event', event.name
            if event.name == FF_ALLOC_EVENT:
                alloc_picture(event.userdata)
            elif event.name == FF_REFRESH_EVENT:
                if self.vs.quit == 0:
                    video_refresh_timer(event.userdata)
            elif event.name == FF_QUIT_EVENT:
                self.vs.quit = 1
                self.stop()
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

        self.update()

        if self.get_width() == -1:
            return

        ret = None
        size = self.get_width() * self.get_height() * 3
        with nogil:
            SDL_LockMutex(vs.pictq_mutex)
        if vs.pixels != NULL:
            ret = PyString_FromStringAndSize(<char *>vs.pixels, size)
        with nogil:
            SDL_UnlockMutex(vs.pictq_mutex)

        return ret


