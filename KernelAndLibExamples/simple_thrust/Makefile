all: hw

hw:
	 nvcc -x cu -O3 -std=c++20 --extended-lambda -arch=sm_86 -I/usr/local/cuda/bin/../include ./simple_thrust.cu -o main -L/usr/local/cuda/bin/../lib64

clean:
	rm -f main

.PHONY: clean hw
