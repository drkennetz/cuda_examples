all: hw

hw:
	 nvcc -x cu -O3 -std=c++20 -arch=sm_86 -I/usr/local/cuda/bin/../include ./matmul_tp_small.cu -o main_small -L/usr/local/cuda/bin/../lib64
	 nvcc -x cu -O3 -std=c++20 -arch=sm_86 -I/usr/local/cuda/bin/../include ./matmul_tp_big.cu -o main_big -L/usr/local/cuda/bin/../lib64
clean:
	rm -f main

.PHONY: clean hw
