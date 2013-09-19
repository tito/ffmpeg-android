#!/bin/bash
# 
# Author: Mathieu Virbel <mat@kivy.org>
# 
# This script compile a special version of ffmpeg only supporting H264 + AAC
# decoding. You can use it if you want to play video into your app, and if you
# can control the initial encoding.
#

usage() {
	echo "Usage:"
	echo ""
	echo "	FFMPEG_ARCHS='x86 armv7a' $0"
	echo ""
	echo "The result will be stored in build/ffmpeg/{x86,armv7a}/"
	echo "Available ARCHS: x86 armv7a"
	echo ""
	echo "Note: to be able to build armv7a, you need the ANDROIDNDK env variable set"
	exit 1
}

if [ "X$FFMPEG_ARCHS" == "X" ]; then
	usage
	exit 1
fi

rm -rf build/ffmpeg
mkdir -p build/ffmpeg
cd ffmpeg

# Don't build any neon version for now
#for version in armv5te armv7a; do
for version in $FFMPEG_ARCHS; do

	echo "==> build for $version"

	DEST=../build/ffmpeg

	FLAGS="--disable-avfilter --disable-everything"
	FLAGS="$FLAGS --enable-parser=h264,aac --enable-decoder=h264,aac"
	FLAGS="$FLAGS --disable-pthreads --enable-protocol=file"
	FLAGS="$FLAGS --enable-demuxer=sdp --enable-pic"
	FLAGS="$FLAGS --enable-small --disable-avdevice"

	# needed to prevent _ffmpeg.so: version node not found for symbol av_init_packet@LIBAVFORMAT_52
	# /usr/bin/ld: failed to set dynamic section sizes: Bad value
	FLAGS="$FLAGS --disable-symver"

	# fix to prevent libavcodec.a(deinterlace.o): relocation R_X86_64_PC32 against symbol
	# `ff_pw_4' can not be used when making a shared object; recompile with -fPIC
	# note: yeap, fPIC is already activated, but doesn't work when compiling shared python.
	# some refs http://www.gentoo.org/proj/en/base/amd64/howtos/index.xml?part=1&chap=3,
	# but no doc found to explain the real issue :/
	#FLAGS="$FLAGS --disable-asm"
	FLAGS="$FLAGS --enable-asm"

	# disable some unused algo
	# note: "golomb" are the one used in our video test, so don't use --disable-golomb
	# note: and for aac decoding: "rdft", "mdct", and "fft" are needed
	FLAGS="$FLAGS --disable-dxva2 --disable-vdpau --disable-vaapi"
	FLAGS="$FLAGS --disable-lpc --disable-huffman --disable-dct --disable-aandct"

	# disable binaries / doc
	FLAGS="$FLAGS --disable-ffmpeg --disable-ffplay --disable-ffprobe --disable-ffserver"
	FLAGS="$FLAGS --disable-doc"

	# why it doesn't work ?
	#FLAGS="$FLAGS --disable-network"

	case "$version" in
		x86)
			EXTRA_CFLAGS=""
			EXTRA_LDLAGS=""
			ABI="x86"
			;;
		neon)
			echo "Arch neon is not supported yet"
			exit 1
			FLAGS="$ARM_FLAGS $FLAGS"
			EXTRA_CFLAGS="-march=armv7-a -mfloat-abi=softfp -mfpu=neon"
			EXTRA_LDFLAGS="-Wl,--fix-cortex-a8"
			# Runtime choosing neon vs non-neon requires
			# renamed files
			ABI="armeabi-v7a"
			;;
		armv7a)
			# in that case, we need ANDROIDNDK
			if [ "X$ANDROIDNDK" == "X" ]; then
				echo "ANDROIDNDK variable not set"
				echo "You must set it to your Android NDK root directory"
				exit 1
			fi

			SYSROOT=$ANDROIDNDK/platforms/android-3/arch-arm
			TOOLCHAIN=`echo $ANDROIDNDK/toolchains/arm-linux-androideabi-4.4.3/prebuilt/*-x86`
			echo "==> toolchain is $TOOLCHAIN"
			export PATH=$TOOLCHAIN/bin:$PATH
			ARM_FLAGS="--target-os=linux --cross-prefix=arm-linux-androideabi- --arch=arm"
			ARM_FLAGS="$ARM_FLAGS --sysroot=$SYSROOT"
			ARM_FLAGS="$ARM_FLAGS --soname-prefix=/data/data/com.bambuser.broadcaster/lib/"

			FLAGS="$ARM_FLAGS $FLAGS"
			FLAGS="$FLAGS --enable-neon"
			#EXTRA_CFLAGS="-march=armv7-a -mfloat-abi=softfp -fPIC -DANDROID"
			EXTRA_CFLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=softfp -fPIC -DANDROID"
			EXTRA_LDFLAGS=""
			ABI="armeabi-v7a"
			;;
		*)
			echo "Unknown platform $version"
			exit 1
			;;
	esac
	DEST="$DEST/$ABI"
	FLAGS="$FLAGS --prefix=$DEST"

	mkdir -p $DEST
	echo $FLAGS --extra-cflags="$EXTRA_CFLAGS" --extra-ldflags="$EXTRA_LDFLAGS" > $DEST/info.txt
	make distclean
	set -x
	./configure $FLAGS --extra-cflags="$EXTRA_CFLAGS" --extra-ldflags="$EXTRA_LDFLAGS" \
		| tee $DEST/configuration.txt
	set +x
	[ $PIPESTATUS == 0 ] || exit 1
	make clean
	make -j4 || exit 1
	make install || exit 1

done

