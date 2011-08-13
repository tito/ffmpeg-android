'''
FFMPEG wrapper for just implementing a Player
=============================================

The player give you access to the raw RGB image (converted if needed.)
The sound is automatically mixed.
'''

__version__ = (1, 0)
__all__ = ('FFVideo', 'FFVideoException')

from _ffmpeg import FFVideo, FFVideoException

if __name__ == '__main__':
    import sys
    from kivy.core.window import Window
    from kivy.uix.image import Image
    from kivy.graphics.texture import Texture
    from kivy.base import runTouchApp
    from kivy.clock import Clock
    from kivy.uix.gridlayout import GridLayout
    from functools import partial

    def queue_frame(img, video, dt):
        frame = video.get_next_frame()
        if frame is None:
            if video.is_open == False:
                img.texture = None
            return
        tex = img.texture
        if tex is None:
            tex = Texture.create(size=(
                video.get_width(), video.get_height()), colorfmt='rgb')
            tex.flip_vertical()
        tex.blit_buffer(frame)
        img.texture = None
        img.texture = tex

    root = GridLayout(cols=2)

    for filename in sys.argv[1:]:
        img = Image()
        root.add_widget(img)
        video = FFVideo(filename)
        Clock.schedule_interval(partial(queue_frame, img, video), 1 / 60.)

        video.open()


    runTouchApp(root)



