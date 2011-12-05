from os.path import dirname, join, realpath
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
if not have_cython:
    # android build ?
    ext_files = [x.replace('.pyx', '.c') for x in ext_files]
    pgs4a_root = environ.get('PGS4A_ROOT')
    if not pgs4a_root:
        raise Exception('This android build must be done inside PGS4A.')
    include_dirs = [
        '../build/ffmpeg/armeabi-v7a/include/',
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
        join(p, 'libavfilter', 'libavfilter.a'),
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
