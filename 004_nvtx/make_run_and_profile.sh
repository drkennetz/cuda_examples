#!/usr/bin/env bash

# I'm not going to make this pretty
make clean
make
rm -rf chk* out*
nsys profile -o chk --stats=true --gpu-metrics-device=all ./main > stdout.dump
nsys stats -f csv -o out chk.sqlite

# remove empties
find . -type f -empty -delete
