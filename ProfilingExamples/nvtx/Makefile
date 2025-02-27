all: hw

hw:
	 nvcc -x cu -O3 -std=c++20 -arch=sm_86 -I/usr/local/cuda/bin/../include ./nvtx_example.cu -o main -L/usr/local/cuda/bin/../lib64 -lnvToolsExt

clean:
	rm -f main

.PHONY: clean hw
