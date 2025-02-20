# Vector Addition using CUDA

## Overview
This repository demonstrates a simple CUDA program that performs element-wise addition of two vectors using GPU acceleration. The implementation uses CUDA kernel functions to parallelize the computation, allocating memory on both the host (CPU) and the device (GPU), and leveraging CUDA APIs for memory transfer and execution control.

## Files
- `vector_add.cu`: The main CUDA source file that implements vector addition.
- `Makefile`: The build script to compile and link the CUDA program.
- `utils.cuh`: A utility header file that likely provides error checking macros (not included in this repository but referenced in `vector_add.cu`).

## Implementation Details
The `vector_add.cu` file contains:
1. **Memory Allocation**: Allocates pinned host memory and GPU memory using `cudaMallocHost` and `cudaMalloc`.
2. **Data Initialization**: Initializes input vectors with constant values.
3. **Memory Transfer**: Copies data from host to device using `cudaMemcpy`.
4. **Kernel Execution**: Calls the `vector_add_kernel` with `8` thread blocks and `256` threads per block.
5. **Memory Transfer Back**: Copies results from the device to the host.
6. **Validation**: Checks correctness using assertions.
7. **Memory Cleanup**: Frees allocated memory on both host and device.

## CUDA Kernel Explanation
```cpp
__global__ void vector_add_kernel(float *d_v1, float *d_v2, float *d_v3, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_v3[i] = d_v1[i] + d_v2[i];
    }
}
```
- This kernel runs in parallel across multiple GPU threads.
- Each thread computes the sum of corresponding elements from `d_v1` and `d_v2`.
- The `if (i < n)` check ensures that out-of-bounds accesses do not occur.
- The total number of threads is determined by the grid (`blockIdx.x`) and block (`threadIdx.x`) indices.

## How to Build and Run
### Prerequisites
Ensure that:
- NVIDIA CUDA Toolkit 12.3 or later is installed.
- Your GPU supports compute capability `sm_86` (e.g., NVIDIA Ampere architecture like A100).

### Compilation
To build the executable, run:
```sh
make
```
This invokes `nvcc` with the following options:
- `-x cu`: Specifies that the source file is CUDA C++.
- `-O3`: Enables compiler optimizations.
- `-std=c++20`: Uses C++20 standard.
- `-arch=sm_86`: Targets GPUs with compute capability 8.6.
- `-I` and `-L`: Include and link CUDA libraries.

### Running the Program
Once compiled, execute the program:
```sh
./main
```
If successful, it will run without errors, performing vector addition and validating results.

### Cleaning Up
To remove the compiled binary, run:
```sh
make clean
```

