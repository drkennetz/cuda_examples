#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

// Move this out to top level.
#ifndef cudaCheckError
#define cudaCheckError(call)                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // cudaCheckError


__global__ void vector_add_kernel(float *d_v1, float *d_v2, float *d_v3, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_v3[i] = d_v1[i] + d_v2[i];
    }
}

// Structure to pass around host memory pointers.
struct Memory {
    float *v1;
    float *v2;
    float *v3;
};

void benchmark(int N, bool usePinnedMemory) {
    size_t size = N * sizeof(float);
    Memory mem;
    
    // Allocate memory
    if (usePinnedMemory) {
        cudaCheckError(cudaMallocHost(&mem.v1, size));
        cudaCheckError(cudaMallocHost(&mem.v2, size));
        cudaCheckError(cudaMallocHost(&mem.v3, size));
    } else {
        mem.v1 = (float*)malloc(size);
        mem.v2 = (float*)malloc(size);
        mem.v3 = (float*)malloc(size);
    }

    // Initialize data
    for (int i = 0; i < N; i++) {
        mem.v1[i] = 1.0f;
        mem.v2[i] = 2.0f;
    }

    // Allocate device memory
    float *d_v1, *d_v2, *d_v3;
    cudaCheckError(cudaMalloc(&d_v1, size));
    cudaCheckError(cudaMalloc(&d_v2, size));
    cudaCheckError(cudaMalloc(&d_v3, size));

    // Create CUDA stream
    cudaStream_t stream;
    cudaCheckError(cudaStreamCreate(&stream));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // Record start time
    cudaCheckError(cudaEventRecord(start, stream));

    // Asynchronous memory transfers
    cudaCheckError(cudaMemcpyAsync(d_v1, mem.v1, size, cudaMemcpyHostToDevice, stream));
    cudaCheckError(cudaMemcpyAsync(d_v2, mem.v2, size, cudaMemcpyHostToDevice, stream));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_v1, d_v2, d_v3, N);

    // Asynchronous memory transfer back to host
    cudaCheckError(cudaMemcpyAsync(mem.v3, d_v3, size, cudaMemcpyDeviceToHost, stream));

    // Record stop time
    cudaCheckError(cudaEventRecord(stop, stream));
    cudaCheckError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate bandwidth
    float bandwidth = 3 * size / (milliseconds / 1000) / 1e9; // GB/s

    std::cout << (usePinnedMemory ? "Pinned" : "Pageable") << " memory, N = " << N 
              << ", Time: " << milliseconds << " ms, Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Clean up
    cudaCheckError(cudaFree(d_v1));
    cudaCheckError(cudaFree(d_v2));
    cudaCheckError(cudaFree(d_v3));
    cudaCheckError(cudaStreamDestroy(stream));
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    if (usePinnedMemory) {
        cudaCheckError(cudaFreeHost(mem.v1));
        cudaCheckError(cudaFreeHost(mem.v2));
        cudaCheckError(cudaFreeHost(mem.v3));
    } else {
        free(mem.v1);
        free(mem.v2);
        free(mem.v3);
    }
}

int main() {
    std::vector<int> sizes = {1024, 1024 * 1024, 10 * 1024 * 1024};
    
    for (int N : sizes) {
        benchmark(N, false); // Pageable memory
        benchmark(N, true);  // Pinned memory
        std::cout << std::endl;
    }

    return 0;
}
