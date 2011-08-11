
ctypedef signed long long int64_t
ctypedef unsigned long long uint64_t
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef short int16_t
ctypedef unsigned short uint16_t

class PixelFormats:
    NONE = -1
    RGB24 = 2

cdef extern from "math.h":
    double fabs(double)
    double exp(double)
    double log(double)

cdef extern from "Python.h":
    object PyString_FromStringAndSize(char *s, Py_ssize_t len)
    void PyEval_InitThreads()

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
    struct AVRational:
        int num
        int den
    struct AVCodec:
        CodecType codec_type
    struct AVFrame:
        unsigned char **data
        int *linesize
        int64_t pts
        int repeat_pict
        void *opaque
    struct AVCodecContext:
        int width
        int height
        int codec_id
        int pix_fmt
        int sample_rate
        int channels
        AVCodec *codec
        CodecType codec_type
        int (*get_buffer)(AVCodecContext *c, AVFrame *pic)
        void (*release_buffer)(AVCodecContext *c, AVFrame *pic)
        AVRational time_base
    struct AVPicture:
        uint8_t *data[4]
        int linesize[4]
    struct AVPacket:
        int stream_index
        int size
        int64_t pts
        int64_t dts
        uint8_t *data
    double av_q2d(AVRational a)
        
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
    void avcodec_flush_buffers(AVCodecContext *avctx)
    void av_init_packet(AVPacket *pkt)


    int avpicture_get_size(int pix_fmt, int width, int height) 
    int avpicture_fill(AVPicture *picture, unsigned char *ptr,
                       int pix_fmt, int width, int height) 
    int avcodec_default_get_buffer(AVCodecContext *s, AVFrame *pic)
    void avcodec_default_release_buffer(AVCodecContext *s, AVFrame *pic)

cdef extern from "libavformat/avformat.h":
    struct AVStream:
        AVCodecContext *codec
        AVRational time_base
    struct ByteIOContext:
        int error
    struct AVFormatContext:
        unsigned int nb_streams
        AVStream **streams
        char filename[1024]
        ByteIOContext *pb
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
    void *av_mallocz(unsigned int size) 
    void av_free(void *ptr) 
    void av_freep(void *ptr)
    int64_t av_gettime()
    void dump_format(AVFormatContext *ic, int index,
            char *url, int is_output) 
    ctypedef int (*URLInterruptCB)()
    void url_set_interrupt_cb(URLInterruptCB)
    int av_seek_frame(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

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

cdef extern from "libavutil/avutil.h":
    int64_t av_rescale_q(int64_t a, AVRational bq, AVRational cq)

cdef extern from "SDL.h":
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

    struct SDL_Thread:
        pass

    SDL_mutex *SDL_CreateMutex() 
    void SDL_DestroyMutex(SDL_mutex *)
    int SDL_LockMutex(SDL_mutex *) nogil
    int SDL_UnlockMutex(SDL_mutex *) nogil

    struct SDL_cond:
        pass

    SDL_cond *SDL_CreateCond()
    void SDL_DestroyCond(SDL_cond *) 
    int SDL_CondSignal(SDL_cond *) nogil
    int SDL_CondWait(SDL_cond *, SDL_mutex *) nogil

    struct SDL_Thread:
        pass

    ctypedef int (*SDLCALL)(void *)
    SDL_Thread *SDL_CreateThread(SDLCALL, void *data) nogil

    char *SDL_GetError()

    struct SDL_UserEvent:
        uint8_t type
        int code
        void *data1
        void *data2

    union SDL_Event:
        uint8_t type

    int SDL_PushEvent(SDL_Event *event)
    void SDL_Delay(int) nogil



