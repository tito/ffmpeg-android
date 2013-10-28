#!/bin/sh

tar -zxvf ffmpeg-1.2.4.tar.gz
for i in `find diffs -type f`; do
	(cd ffmpeg && patch -p1 < ../$i)
done
