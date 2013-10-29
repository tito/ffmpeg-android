
ctypedef signed long long int64_t
ctypedef unsigned long long uint64_t
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef short int16_t
ctypedef unsigned short uint16_t

cdef extern from *:
    ctypedef char* const_char_ptr "const char *"
    ctypedef uint8_t* const_uint8_ptr "const uint8_t * const*"


DEF SDL_INIT_AUDIO = 0x10
DEF AVSEEK_FLAG_BACKWARD = 1
DEF AV_TIME_BASE = 1000000.

cdef extern from "Python.h":
    void PyEval_InitThreads()

cdef extern from "libavcodec/avcodec.h" nogil:
    enum AVMediaType:
        AVMEDIA_TYPE_UNKNOWN = -1
        AVMEDIA_TYPE_VIDEO = 0
        AVMEDIA_TYPE_AUDIO = 1
        AVMEDIA_TYPE_DATA = 2
        AVMEDIA_TYPE_SUBTITLE = 3
    struct AVRational:
        int num
        int den
    struct AVCodec:
        AVMediaType codec_type
    struct AVFrame:
        unsigned char **data
        int *linesize
        int64_t pts
        int repeat_pict
        int nb_samples
        int format
        uint64_t channel_layout
        void *opaque
        int sample_rate

    enum AVSampleFormat:
        pass
    struct AVCodecContext:
        int width
        int height
        int codec_id
        int pix_fmt
        int sample_rate
        int channels
        AVCodec *codec
        AVMediaType codec_type
        int (*get_buffer)(AVCodecContext *c, AVFrame *pic)
        void (*release_buffer)(AVCodecContext *c, AVFrame *pic)
        AVRational time_base
        AVSampleFormat sample_fmt
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

    int avcodec_open2(AVCodecContext *avctx, AVCodec *codec,
            AVDictionary **options)
    int avcodec_close(AVCodecContext *avctx)

    enum AVLockOp:
        AV_LOCK_CREATE
        AV_LOCK_OBTAIN
        AV_LOCK_RELEASE
        AV_LOCK_DESTROY
    ctypedef int (*lockmgr_t)(void **mutex, AVLockOp op)
    int av_lockmgr_register(lockmgr_t cb)

    AVFrame *avcodec_alloc_frame() 
    int avcodec_decode_video2(AVCodecContext *avctx, AVFrame *picture,
                         int *got_picture_ptr, AVPacket *avpkt)
    int avcodec_decode_audio4(AVCodecContext *avctx, AVFrame *frame, int
            *got_frame_ptr, const AVPacket *avpkt)
    void av_free_packet(AVPacket *pkt) nogil
    int av_dup_packet(AVPacket *pkt) nogil
    void avcodec_flush_buffers(AVCodecContext *avctx)
    void av_init_packet(AVPacket *pkt)


    int avpicture_get_size(int pix_fmt, int width, int height) 
    int avpicture_fill(AVPicture *picture, unsigned char *ptr,
                       int pix_fmt, int width, int height) 
    int avcodec_default_get_buffer(AVCodecContext *s, AVFrame *pic)
    void avcodec_default_release_buffer(AVCodecContext *s, AVFrame *pic)

    void avcodec_get_frame_defaults(AVFrame *)

cdef extern from "libavresample/avresample.h" nogil:
    struct ResampleContext:
        pass
    struct AVAudioResampleContext:
        pass

    void avresample_free(AVAudioResampleContext **)
    int avresample_open(AVAudioResampleContext *)
    void avresample_close(AVAudioResampleContext *)
    AVAudioResampleContext *avresample_alloc_context()
    int avresample_convert(AVAudioResampleContext *avr, uint8_t **output,
                       int out_plane_size, int out_samples, uint8_t **input,
                       int in_plane_size, int in_samples)
    int avresample_available(AVAudioResampleContext *avr)
    int avresample_read(AVAudioResampleContext *avr, void **output, int
            nb_samples)

cdef extern from "libavformat/avformat.h" nogil:
    ctypedef int (*URLInterruptCB)(void*)
    struct AVStream:
        AVCodecContext *codec
        AVRational time_base
        int64_t duration
    struct ByteIOContext:
        int error
    struct AVIOInterruptCB:
        URLInterruptCB callback
        void *opaque
    struct AVFormatContext:
        unsigned int nb_streams
        AVStream **streams
        char filename[1024]
        ByteIOContext *pb
        int64_t duration
        AVIOInterruptCB interrupt_callback
    struct AVFormatParameters:
        pass
    struct AVInputFormat:
        pass
    struct AVPacketList:
        AVPacket pkt
        AVPacketList *next
    struct AVDictionary:
        pass

    # cannot work in threads mode, use GIL to act as a mutex
    AVFormatContext *avformat_alloc_context()
    int avformat_open_input(AVFormatContext **ic, char *filename,
            AVInputFormat *fmt, AVDictionary **options) with gil
    int avformat_find_stream_info(AVFormatContext *ic, AVDictionary **options)
    int av_read_frame(AVFormatContext *s, AVPacket *pkt) 
    void av_close_input_file(AVFormatContext *s) 

    void *av_malloc(unsigned int size) nogil
    void *av_mallocz(unsigned int size) nogil
    void av_free(void *ptr)  nogil
    void av_freep(void *ptr) nogil
    int64_t av_gettime()
    void av_dump_format(AVFormatContext *ic, int index,
            char *url, int is_output) 
    int av_seek_frame(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

cdef extern from "libswscale/swscale.h" nogil:
    struct SwsContext:
        pass
    struct SwsFilter:
        pass
    int sws_scale(SwsContext *context, const_uint8_ptr srcSlice,
            int srcStride[], int srcSliceY, int srcSliceH,
            unsigned char* dst[], int dstStride[]) 
    SwsContext *sws_getContext(int srcW, int srcH, int srcFormat,
            int dstW, int dstH, int dstFormat, int flags,
            SwsFilter *srcFilter, SwsFilter *dstFilter, double *param) 

cdef extern from "libavutil/avutil.h" nogil:
    void *av_realloc(void *ptr, size_t size)
    int av_get_bytes_per_sample(AVSampleFormat sample_fmt)
    int64_t av_rescale_q(int64_t a, AVRational bq, AVRational cq)
    int64_t av_get_default_channel_layout(int)
    int av_samples_get_buffer_size(int *linesize, int nb_channels, int nb_samples,
        AVSampleFormat sample_fmt, int align)

cdef extern from "libavutil/opt.h" nogil:
    int av_opt_set_int(void *, const_char_ptr, int64_t, int)

cdef extern from "libavutil/pixfmt.h" nogil:
    int AV_PIX_FMT_RGB24
    #int AV_PIX_FMT_RGBA

cdef extern from "SDL.h" nogil:
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

    struct SDL_mutex:
        pass

    struct SDL_Thread:
        pass

    SDL_mutex *SDL_CreateMutex()
    void SDL_DestroyMutex(SDL_mutex *)
    int SDL_LockMutex(SDL_mutex *)
    int SDL_UnlockMutex(SDL_mutex *)

    struct SDL_cond:
        pass

    SDL_cond *SDL_CreateCond()
    void SDL_DestroyCond(SDL_cond *)
    int SDL_CondSignal(SDL_cond *)
    int SDL_CondWait(SDL_cond *, SDL_mutex *)

    struct SDL_Thread:
        pass

    ctypedef int (*SDLCALL)(void *)
    SDL_Thread *SDL_CreateThread(SDLCALL, void *data)
    void SDL_WaitThread(SDL_Thread *thread, int *status)
    uint32_t SDL_ThreadID()

    char *SDL_GetError()

    struct SDL_UserEvent:
        uint8_t type
        int code
        void *data1
        void *data2

    union SDL_Event:
        uint8_t type

    int SDL_PushEvent(SDL_Event *event)
    void SDL_Delay(int)
    int SDL_Init(int)
    void SDL_LockAudio()
    void SDL_UnlockAudio()

cdef extern from "SDL_mixer.h" nogil:
    struct Mix_Chunk:
        pass
    int Mix_Init(int)
    int Mix_OpenAudio(int frequency, uint16_t format, int channels, int chunksize)
    void Mix_Pause(int channel)
    void Mix_Resume(int channel)
    void Mix_CloseAudio()
    int Mix_PlayChannel(int channel, Mix_Chunk *chunk, int loops)
    int Mix_HaltChannel(int channel)
    ctypedef void (*Mix_EffectFunc_t)(int, void *, int, void *)
    ctypedef void (*Mix_EffectDone_t)(int, void *)
    int Mix_RegisterEffect(int chan, Mix_EffectFunc_t f, Mix_EffectDone_t d, void * arg)
    int Mix_UnregisterAllEffects(int chan)
    int Mix_AllocateChannels(int numchans)
    Mix_Chunk * Mix_LoadWAV(char *filename)
    int Mix_QuerySpec(int *frequency,uint16_t *format,int *channels)
    int Mix_Volume(int chan, int volume)


cdef extern from "libavutil/samplefmt.h" nogil:
    int AV_SAMPLE_FMT_U8P
    int AV_SAMPLE_FMT_S16
    int AV_SAMPLE_FMT_S16P

cdef extern from "libavutil/channel_layout.h" nogil:
    int AV_CH_LAYOUT_STEREO

cdef extern from "stdarg.h" nogil:
    ctypedef struct va_list:
        pass

cdef extern from "libavutil/log.h" nogil:
    ctypedef void (*log_callback_t)(void *, int, const_char_ptr, va_list)
    void av_log_set_callback(log_callback_t cb)
    void av_log_format_line(void *, int, const_char_ptr, va_list, char *, int,
            int*)
