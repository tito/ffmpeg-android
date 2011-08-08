'''
Python library for decoding video
=================================

References:
    http://bambuser.com/opensource
    https://github.com/tranx/pyffmpeg/blob/master/pyffmpeg.pyx
    http://dranger.com/ffmpeg/
'''
##
## usefull constants
##

ctypedef signed long long int64_t
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef short int16_t
ctypedef unsigned short uint16_t

class PixelFormats:
    NONE = -1
    RGB24 = 2

cdef int AUDIO_S16SYS = 0x8010

cdef extern from "Python.h":
    object PyString_FromStringAndSize(char *s, Py_ssize_t len)

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr) 
    void *malloc(size_t) 

cdef extern from "string.h":
    void *memcpy(void *dest, void *src, size_t n) 
    void *memset(void *s, int c, size_t n) 

cdef extern from "libavcodec/avcodec.h":
    enum CodecType:
        CODEC_TYPE_UNKNOWN = -1
        CODEC_TYPE_VIDEO = 0
        CODEC_TYPE_AUDIO = 1
        CODEC_TYPE_DATA = 2
        CODEC_TYPE_SUBTITLE = 3
    struct AVCodec:
        CodecType codec_type
    struct AVCodecContext:
        int width
        int height
        int codec_id
        int pix_fmt
        int sample_rate
        int channels
        AVCodec *codec
        CodecType codec_type
    struct AVFrame:
        unsigned char **data
        int *linesize
        int64_t pts
    struct AVPicture:
        pass
    struct AVPacket:
        int stream_index
        int size
        int64_t pts
        int64_t dts
        uint8_t *data
        
    void av_register_all() 
    AVCodec *avcodec_find_decoder(int cid) 
    int avcodec_open(AVCodecContext *avctx, AVCodec *codec) 
    AVFrame *avcodec_alloc_frame() 
    int avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,
                         int *got_picture_ptr, AVPacket *avpkt)
    int avcodec_decode_audio2(AVCodecContext *avctx, int16_t *samples,
                        int *frame_size_ptr, uint8_t *buf, int buf_size) 
    int avcodec_close(AVCodecContext *avctx) 
    void av_free_packet(AVPacket *pkt) 
    int av_dup_packet(AVPacket *pkt) 


    int avpicture_get_size(int pix_fmt, int width, int height) 
    int avpicture_fill(AVPicture *picture, unsigned char *ptr,
                       int pix_fmt, int width, int height) 

cdef extern from "libavformat/avformat.h":
    struct AVStream:
        AVCodecContext *codec
    struct AVFormatContext:
        unsigned int nb_streams
        AVStream **streams
    struct AVFormatParameters:
        pass
    struct AVInputFormat:
        pass
    struct AVPacketList:
        AVPacket pkt
        AVPacketList *next

    int av_open_input_file(AVFormatContext **ic, char *filename,
            AVInputFormat *fmt, int buf_size, AVFormatParameters *ap) 
    int av_find_stream_info(AVFormatContext *ic) 
    int av_read_frame(AVFormatContext *s, AVPacket *pkt) 
    void av_close_input_file(AVFormatContext *s) 

    void *av_malloc(unsigned int size) 
    void av_free(void *ptr) 
    void dump_format(AVFormatContext *ic, int index,
            char *url, int is_output) 
    int AVCODEC_MAX_AUDIO_FRAME_SIZE = 192000

cdef extern from "libswscale/swscale.h":
    struct SwsContext:
        pass
    struct SwsFilter:
        pass
    int sws_scale(SwsContext *context, unsigned char* srcSlice[],
            int srcStride[], int srcSliceY, int srcSliceH,
            unsigned char* dst[], int dstStride[]) 
    SwsContext *sws_getContext(int srcW, int srcH, int srcFormat,
            int dstW, int dstH, int dstFormat, int flags,
            SwsFilter *srcFilter, SwsFilter *dstFilter, double *param) 

cdef extern from "SDL/SDL.h":
    struct SDL_AudioSpec:
        int freq
        uint16_t format
        uint8_t channels
        uint8_t silence
        uint16_t samples
        uint16_t padding
        uint32_t size
        void (*callback)(void *userdata, uint8_t *stream, int len)
        void *userdata

    int SDL_OpenAudio(SDL_AudioSpec *desired, SDL_AudioSpec *obtained)
    void SDL_PauseAudio(int pause_on)

    struct SDL_mutex:
        pass

    SDL_mutex *SDL_CreateMutex() 
    void SDL_DestroyMutex(SDL_mutex *)
    int SDL_LockMutex(SDL_mutex *) nogil
    int SDL_UnlockMutex(SDL_mutex *)

    struct SDL_cond:
        pass

    SDL_cond *SDL_CreateCond()
    void SDL_DestroyCond(SDL_cond *) 
    int SDL_CondSignal(SDL_cond *) nogil
    int SDL_CondWait(SDL_cond *, SDL_mutex *) nogil



cdef int g_have_register = 0

cdef struct PacketQueue:
    AVPacketList *first_pkt, *last_pkt
    int nb_packets
    int size
    SDL_mutex *mutex
    SDL_cond *cond

cdef struct AudioCtx:
    PacketQueue q
    int quit
    AVCodecContext *codec_ctx
    unsigned int audio_buf_size
    unsigned int audio_buf_index
    unsigned int audio_buf_maxsize
    uint8_t *audio_buf

cdef int audio_decode_frame(AudioCtx *ctx):
    cdef AVPacket pkt
    cdef uint8_t *audio_pkt_data = NULL
    cdef int audio_pkt_size = 0
    cdef int len1, data_size
    cdef AVCodecContext *audio_codec_ctx = ctx.codec_ctx

    pkt.data = NULL


    while True:

        while audio_pkt_size > 0:

            data_size = ctx.audio_buf_maxsize

            len1 = avcodec_decode_audio2(
                    audio_codec_ctx, <int16_t *>ctx.audio_buf, &data_size, 
                    audio_pkt_data, audio_pkt_size)

            if len1 < 0:
                # if error, skip frame
                audio_pkt_size = 0
                break
            audio_pkt_data += len1
            audio_pkt_size -= len1
            if data_size <= 0:
                # No data yet, get more frames
                continue
            # We have data, return it and come back for more later */
            return data_size

        if pkt.data:
            av_free_packet(&pkt)

        if ctx.quit:
            return -1

        if audio_queue_get(ctx, &pkt, 1) < 0:
            return -1

        audio_pkt_data = pkt.data
        audio_pkt_size = pkt.size

    return 0


cdef int audio_queue_get(AudioCtx *ctx, AVPacket *pkt, int block):
    cdef AVPacketList *pkt1
    cdef PacketQueue *q = &ctx.q
    cdef int ret = -1

    with nogil:
        SDL_LockMutex(q.mutex)

    while True:

        if ctx.quit:
            ret = -1
            break

        pkt1 = q.first_pkt
        if pkt1 != NULL:
            q.first_pkt = pkt1.next
            if q.first_pkt == NULL:
                q.last_pkt = NULL
            q.nb_packets -= 1
            print 'left', q.nb_packets
            q.size -= pkt1.pkt.size
            memcpy(pkt, &pkt1.pkt, sizeof(AVPacket))
            av_free(pkt1)
            ret = 1
            break
        elif block == 0:
            ret = 0
            break
        else:
            with nogil:
                SDL_CondWait(q.cond, q.mutex)

    SDL_UnlockMutex(q.mutex)

    return ret

cdef int audio_queue_put(AudioCtx *ctx, AVPacket *pkt) :
    cdef AVPacketList *pkt1
    cdef PacketQueue *q = &ctx.q

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

cdef void audio_callback(void *userdata, unsigned char *stream, int l) with gil:

    cdef AudioCtx *ctx = <AudioCtx *>userdata
    cdef int len1, audio_size

    while l > 0:

        if ctx.audio_buf_index >= ctx.audio_buf_size:
            # We have already sent all our data; get more
            audio_size = audio_decode_frame(ctx)
            if audio_size < 0:
                # If error, output silence
                ctx.audio_buf_size = 1024
                memset(ctx.audio_buf, 0, ctx.audio_buf_size)
            else:
                ctx.audio_buf_size = audio_size
            ctx.audio_buf_index = 0

        len1 = ctx.audio_buf_size - ctx.audio_buf_index
        if len1 > l:
            len1 = l
        memcpy(stream, <uint8_t *>ctx.audio_buf + ctx.audio_buf_index, len1)
        l -= len1
        stream += len1
        ctx.audio_buf_index += len1
        

class FFVideoException(Exception):
    pass

cdef class FFVideo:
    cdef AVFrame *ff_frame, *ff_frame_rgb
    cdef AVCodec *ff_video_codec, *ff_audio_codec
    cdef AVFormatContext *ff_format_ctx
    cdef AVCodecContext *ff_video_codec_ctx, *ff_audio_codec_ctx
    cdef SwsContext *ff_sw_ctx
    cdef int ff_video_stream, ff_audio_stream
    cdef int ff_data_size
    cdef unsigned char *ff_data
    cdef unsigned char  *pixels
    cdef bytes filename
    cdef int is_opened
    cdef int quit
    cdef int audio_queue_size
    cdef AudioCtx audio_ctx

    def __cinit__(self, filename):
        self.quit = 0
        self.filename = None
        self.ff_format_ctx = NULL
        self.ff_video_stream = -1
        self.ff_audio_stream = -1
        self.ff_video_codec = NULL
        self.ff_audio_codec = NULL
        self.ff_video_codec_ctx = NULL
        self.ff_audio_codec_ctx = NULL
        self.ff_frame = NULL
        self.ff_frame_rgb = NULL
        self.ff_data = NULL
        self.ff_data_size = -1
        self.ff_sw_ctx = NULL
        self.pixels = NULL
        self.is_opened = 0
        memset(&self.audio_ctx, 0, sizeof(self.audio_ctx))
        self.audio_ctx.q.mutex = SDL_CreateMutex()
        self.audio_ctx.q.cond = SDL_CreateCond()

    def __init__(self, filename):
        self.filename = filename
        self.open()

    cdef void open(self):
        cdef int i
        global g_have_register

        # ensure that ffmpeg have been registered first
        if g_have_register == 0:
            av_register_all()
            g_have_register = 1

        if av_open_input_file(&self.ff_format_ctx, self.filename, NULL, 0, NULL) != 0:
            raise FFVideoException('Unable to open input file')

        self.is_opened = 1

        if av_find_stream_info(self.ff_format_ctx) < 0:
            raise FFVideoException('Unable to find stream info')

        # found at least one video and audio stream
        for i in xrange(self.ff_format_ctx.nb_streams):
            if self.ff_format_ctx.streams[i].codec.codec_type == \
                CODEC_TYPE_VIDEO and self.ff_video_stream < 0:
                self.ff_video_stream = i
            if self.ff_format_ctx.streams[i].codec.codec_type == \
                CODEC_TYPE_AUDIO and self.ff_audio_stream < 0:
                self.ff_audio_stream = i

        #
        # video part
        #

        if self.ff_video_stream == -1:
            raise FFVideoException('Unable to found video stream')

        self.ff_video_codec_ctx = \
            self.ff_format_ctx.streams[self.ff_video_stream].codec
        
        # find decoder for video stream
        self.ff_video_codec = avcodec_find_decoder(self.ff_video_codec_ctx.codec_id)
        if self.ff_video_codec == NULL:
            raise FFVideoException('Unable to found decoder for video stream')

        # open video codec
        if avcodec_open(self.ff_video_codec_ctx, self.ff_video_codec) < 0:
            raise FFVideoException('Unable to open decoder for video stream')

        # alloc frame
        self.ff_frame = avcodec_alloc_frame()
        if self.ff_frame == NULL:
            raise FFVideoException('Unable to allocate codec frame (raw)')
        self.ff_frame_rgb = avcodec_alloc_frame()
        if self.ff_frame_rgb == NULL:
            raise FFVideoException('Unable to allocate codec frame (rgb)')

        # determine required buffer size and allocate
        self.ff_data_size = avpicture_get_size(PixelFormats.RGB24,
                self.ff_video_codec_ctx.width, self.ff_video_codec_ctx.height)
        self.ff_data = <unsigned char *>av_malloc(self.ff_data_size * sizeof(unsigned char))

        # assign appropriate parts of buffer to image planes
        avpicture_fill(<AVPicture *>self.ff_frame_rgb, self.ff_data, PixelFormats.RGB24,
                self.ff_video_codec_ctx.width, self.ff_video_codec_ctx.height)

        #
        # Audio part
        #

        # don't go further if we don't have audio
        if self.ff_audio_stream == -1:
            return

        self.ff_audio_codec_ctx = \
            self.ff_format_ctx.streams[self.ff_audio_stream].codec

        # find decoder for audio stream
        self.ff_audio_codec = avcodec_find_decoder(self.ff_audio_codec_ctx.codec_id)
        if self.ff_audio_codec == NULL:
            raise FFVideoException('Unable to found decoder for audio stream')
        print 'Audio codec id:', self.ff_audio_codec_ctx.codec_id

        # open audio codec
        if avcodec_open(self.ff_audio_codec_ctx, self.ff_audio_codec) < 0:
            raise FFVideoException('Unable to open decoder for audio stream')

        #
        # Init audio part
        #
        self.audio_ctx.codec_ctx = self.ff_audio_codec_ctx
        self.audio_ctx.audio_buf_maxsize = 192000 * 3 / 2
        self.audio_ctx.audio_buf = <uint8_t *>av_malloc(
                self.audio_ctx.audio_buf_maxsize + AVCODEC_MAX_AUDIO_FRAME_SIZE)

        cdef SDL_AudioSpec wanted_spec, spec
        wanted_spec.freq = self.ff_audio_codec_ctx.sample_rate
        wanted_spec.format = AUDIO_S16SYS
        wanted_spec.channels = self.ff_audio_codec_ctx.channels
        wanted_spec.silence = 0
        wanted_spec.samples = 1024
        wanted_spec.callback = audio_callback
        wanted_spec.userdata = &self.audio_ctx

        if SDL_OpenAudio(&wanted_spec, &spec) < 0:
            raise FFVideoException('Unable to initialize SDL audio')

        SDL_PauseAudio(0)
            


    cdef void update(self):
        cdef int got_frame
        cdef AVPacket packet
        while av_read_frame(self.ff_format_ctx, &packet) >= 0:

            got_frame = 0

            # video stream packet ?
            if packet.stream_index == self.ff_video_stream:
                # decode video frame
                avcodec_decode_video2(self.ff_video_codec_ctx, self.ff_frame,
                        &got_frame, &packet)

                if got_frame:

                    # first time, init swscale context
                    if self.ff_sw_ctx == NULL:
                        self.ff_sw_ctx = sws_getContext(
                            self.ff_video_codec_ctx.width,
                            self.ff_video_codec_ctx.height,
                            self.ff_video_codec_ctx.pix_fmt,
                            self.ff_video_codec_ctx.width,
                            self.ff_video_codec_ctx.height,
                            PixelFormats.RGB24, 4, #SWS_BICUBIC
                            NULL, NULL, NULL)
                        if self.ff_sw_ctx == NULL:
                            raise FFVideoException('Unable to initialize conversion context')

                    # convert the image from native to RGB
                    sws_scale(self.ff_sw_ctx, self.ff_frame.data,
                            self.ff_frame.linesize, 0,
                            self.ff_video_codec_ctx.height,
                            self.ff_frame_rgb.data,
                            self.ff_frame_rgb.linesize)

            # audio stream packet ?
            elif packet.stream_index == self.ff_audio_stream:
                # queue the audio packet
                audio_queue_put(&self.audio_ctx, &packet)
                continue

            # free the packet
            av_free_packet(&packet)

            if got_frame:
                break

    cpdef int get_width(self):
        return self.ff_video_codec_ctx.width

    cpdef int get_height(self):
        return self.ff_video_codec_ctx.height

    def get_next_frame(self):
        cdef int size, y, index
        cdef int width = self.get_width()
        cdef int height = self.get_height()
        size = width * height * 3
        if self.pixels == NULL:
            self.pixels = <unsigned char *>malloc(size * sizeof(unsigned char))
        if self.pixels == NULL:
            raise FFVideoException('Unable to allocate memory for frame')

        # do one update
        self.update()

        # copy frame into pixels
        index = 0
        for y in xrange(self.get_height()):
            memcpy(&self.pixels[index], self.ff_frame_rgb.data[0] + \
                    y * self.ff_frame_rgb.linesize[0], width * 3)
            index += width * 3

        return PyString_FromStringAndSize(<char *>self.pixels, size)

