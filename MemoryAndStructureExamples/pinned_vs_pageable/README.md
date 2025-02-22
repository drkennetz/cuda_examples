# Pinned Vs Pagged Type of Host Memory Model

## Overview
This repository demonstrates a CUDA program that compares the performance of pinned (page-locked) and pageable memory for host-to-device and device-to-host memory transfers. Host being CPU and device being GPU. The program benchmarks memory transfer times and bandwidth for different vector sizes using CUDA streams and asynchronous memory copy operations.

## Files
- `pinned_vs_pageable.cu`: The main CUDA source file that implements the benchmarking of pinned vs pageable memory.
- `Makefile`: The build script to compile and link the CUDA program.
- `utils.cuh`: A utility header file providing error checking macros (not included in this repository but referenced in `pinned_vs_pageable.cu`).

## Implementation Details
The `pinned_vs_pageable.cu` file contains:
1. **Memory Allocation**: 
    - Allocates either pinned memory using `cudaMallocHost` or pageable memory using `malloc`.
    - Allocates GPU memory using `CudaMalloc`.
2. **Data Initialization**: Initializes input vectors with predefined constant values.
3. **CUDA Stream Creation**: Creates the CUDA stream to enable asynchronous execution using `cudaStreamCreate`.
3. **Memory Transfer**: Copies data from host to device using `cudaMemcpyAsync`.
4. **Kernel Execution**: Calls the `vector_add_kernel` with dynamic number of blocks and threads.
5. **Memory Transfer Back**: Copies computed results asynchronously back to the host.
6. **Timing and Benchmarking**: Uses CUDA events like `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize` and `cudaEventElapsedTime` to measure execution time and compute memory bandwidth.
7. **Memory Cleanup**: Frees both host and device memory and destroys CUDA streams and events.

## CUDA Kernel Explanation
```cpp
__global__ void vector_add_kernel(float *d_v1, float *d_v2, float *d_v3, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_v3[i] = d_v1[i] + d_v2[i];
    }
}
```
- This kernel executes element-wise addition of two input vectors running in parallel across multiple GPU threads.
- Each thread computes the sum of corresponding elements from `d_v1` and `d_v2`.
- The `if (i < n)` check ensures that out-of-bounds accesses do not occur.
- The total number of threads is determined by the grid (`blockIdx.x`) and block (`threadIdx.x`) indices.

## Benchmarking Methodology
The program benchmarks the performance of memory tranfers for different vector sizes:
- 4KB (1024 elements)
- 4MB (1M elements) 
- 40MB (10M elements)

For each size, the program measures execution time and calculates bandwidth (in GB/s), comparing pinned and pageable memory performance.

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
