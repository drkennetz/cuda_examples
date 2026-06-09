all: hw

# Targets sm_90 (H100). Change -arch to match your GPU:
#   A100 = sm_80, RTX 30xx/A10 = sm_86, RTX 40xx = sm_89, H100 = sm_90.
hw:
	 nvcc -x cu -O3 -std=c++20 -arch=sm_90 -I/usr/local/cuda/bin/../include ./attention.cu -o main -L/usr/local/cuda/bin/../lib64

clean:
	rm -f main

.PHONY: clean hw
