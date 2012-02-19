from os.path import join, realpath
from os import environ
from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    have_cython = True
    cmdclass = { 'build_ext': build_ext }
except ImportError:
    have_cython = False
    cmdclass = {}


libraries = ['avcodec', 'avformat', 'swscale', 'SDL', 'SDL_mixer']
library_dirs = []
include_dirs = []
extra_objects = []
extra_compile_args=['-ggdb', '-O0']

ext_files = ['ffmpeg/_ffmpeg.pyx']
root_ffmpeg = environ.get('FFMPEG_ROOT')
if root_ffmpeg:
    if not have_cython:
        ext_files = [x.replace('.pyx', '.c') for x in ext_files]
    root_ffmpeg = realpath(root_ffmpeg)
    include_dirs = [join(root_ffmpeg, 'include')]
    if environ.get('FFMPEG_INCLUDES'):
        include_dirs += environ.get('FFMPEG_INCLUDES').split(' ')
    if environ.get('FFMPEG_LIBRARY_DIRS'):
        library_dirs += environ.get('FFMPEG_LIBRARY_DIRS').split(' ')
    libraries = environ.get('FFMPEG_LIBRARIES', 'gcc z sdl sdl_mixer m').split(' ')
    extra_compile_args = ['-ggdb', '-O3']
    p = join(root_ffmpeg, 'lib')
    extra_objects = [
        join(p, 'libavformat.a'),
        join(p, 'libavcodec.a'),
        join(p, 'libswscale.a'),
        join(p, 'libavcore.a'),
        join(p, 'libavutil.a')]

elif not have_cython:
    # Special hack for PGS4A-android, should we deprecated it ?
    ext_files = [x.replace('.pyx', '.c') for x in ext_files]
    pgs4a_root = environ.get('PGS4A_ROOT')
    if not pgs4a_root:
        raise Exception('This android build must be done inside PGS4A.')
    include_dirs = ['../build/ffmpeg/armeabi-v7a/include/']
    e=[
        join(pgs4a_root, 'jni', 'sdl', 'include'),
        join(pgs4a_root, 'jni', 'sdl_mixer')
    ]
    p = realpath('../ffmpeg/')
    libraries = ['gcc', 'z', 'sdl', 'sdl_mixer']
    extra_objects = [
        join(p, 'libavcodec', 'libavcodec.a'),
        join(p, 'libavformat', 'libavformat.a'),
        join(p, 'libavcodec', 'libavcodec.a'),
        join(p, 'libavdevice', 'libavdevice.a'),
        #join(p, 'libavfilter', 'libavfilter.a'),
        join(p, 'libswscale', 'libswscale.a'),
        join(p, 'libavcore', 'libavcore.a'),
        join(p, 'libavutil', 'libavutil.a'),
        #join(p, 'libpostproc', 'libavpostproc.a'),
        ]
else:
    include_dirs.append('/usr/include/SDL')

ext = Extension(
    'ffmpeg._ffmpeg',
    ext_files,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args,
)

setup(
    name='ffmpeg',
    version='1.0',
    author='Mathieu Virbel',
    author_email='mat@kivy.org',
    url='http://txzone.net/',
    license='LGPL',
    description='A Python wrapper around ffmpeg for decoding video/audio.',
    ext_modules=[ext],
    cmdclass=cmdclass,
    packages=['ffmpeg'],
    package_dir={'ffmpeg': 'ffmpeg'},
    package_data={'ffmpeg': ['*.wav']}
)
