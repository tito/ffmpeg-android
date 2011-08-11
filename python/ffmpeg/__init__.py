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
    video.open()
    if len(sys.argv) > 2:
        tex2 = None
        video2 = FFVideo(sys.argv[2])
        video2.open()

    img = Image()
    img2 = Image()

    def queue_frame(dt):
        print '==== queue frame asked.'
        global tex
        frame = video.get_next_frame()
        if frame is not None:
            if tex is None:
                tex = Texture.create(size=(
                    video.get_width(), video.get_height()), colorfmt='rgb')
                tex.flip_vertical()
            tex.blit_buffer(frame)
            img.texture = None
            img.texture = tex

        if len(sys.argv) > 2:
            global tex2
            frame = video2.get_next_frame()
            if frame is not None:
                if tex2 is None:
                    tex2 = Texture.create(size=(
                        video2.get_width(), video2.get_height()), colorfmt='rgb')
                    tex2.flip_vertical()
                tex2.blit_buffer(frame)
                img2.texture = None
                img2.texture = tex2

    print 'schedule it'
    Clock.schedule_interval(queue_frame, 1 / 60.)

    print 'run it'
    from kivy.uix.boxlayout import BoxLayout
    root = BoxLayout()
    root.add_widget(img)
    if len(sys.argv) > 2:
        root.add_widget(img2)
    runTouchApp(root)



