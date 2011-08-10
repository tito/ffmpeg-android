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

    tex = None
    video = FFVideo(sys.argv[1])

    img = Image()
    img.texture = tex

    def queue_frame(dt):
        print '==== queue frame asked.'
        global tex
        frame = video.get_next_frame()
        if frame is None:
            return
        if tex is None:
            tex = Texture.create(size=(
                video.get_width(), video.get_height()), colorfmt='rgb')
            tex.flip_vertical()
        tex.blit_buffer(frame)
        img.texture = None
        img.texture = tex

    print 'schedule it'
    Clock.schedule_interval(queue_frame, 1 / 60.)

    print 'run it'
    runTouchApp(img)



