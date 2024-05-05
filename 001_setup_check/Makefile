all: hw

hw:
	 nvcc -x cu -O3 -std=c++11 -arch=sm_86 -I/usr/local/cuda-12.3/bin/../include ./hello_world.cu -o hello_world -L/usr/local/cuda-12.3/bin/../lib64

clean:
	rm -f hello_world

.PHONY: clean hw
