#!/usr/bin/env bash

EXECUTABLE=${1:-main}

# I'm not going to make this pretty
make
rm -rf chk* out*
nsys profile -o chk --stats=true --gpu-metrics-device=all ./${EXECUTABLE}
nsys stats -f csv -o out chk.sqlite

# remove empties
find . -type f -empty -delete
