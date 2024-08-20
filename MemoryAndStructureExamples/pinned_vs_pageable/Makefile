all: hw

hw:
	 nvcc -x cu -O3 -std=c++11 -arch=sm_86 -I/usr/local/cuda-12.3/bin/../include ./pinned_vs_pageable.cu -o main -L/usr/local/cuda-12.3/bin/../lib64

clean:
	rm -f main

.PHONY: clean hw
