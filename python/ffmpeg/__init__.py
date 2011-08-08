__version__ = '1.0'
__all__ = ('FFVideo', 'FFVideoException')

from _ffmpeg import FFVideo, FFVideoException

if __name__ == '__main__':
    import sys
    from kivy.core.window import Window
    from kivy.uix.image import Image
    from kivy.graphics.texture import Texture
    from kivy.base import runTouchApp
    from kivy.clock import Clock

    print 'before ffvideo'
    video = FFVideo(sys.argv[1])
    print 'after ffvideo'

    tex = Texture.create(size=(video.get_width(), video.get_height()), colorfmt='rgb')
    tex.flip_vertical()
    img = Image()
    img.texture = tex

    print 'create texture'

    def queue_frame(dt):
        frame = video.get_next_frame()
        tex.blit_buffer(frame)
        img.texture = None
        img.texture = tex

    print 'schedule it'
    Clock.schedule_interval(queue_frame, 0)

    print 'run it'
    runTouchApp(img)



