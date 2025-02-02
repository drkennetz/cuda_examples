#include <stdio.h>
#include <cuda_runtime.h>
#include "../../utils/utils.cuh"

#define N (1 << 20)  // 1M elements
#define THREADS 1024  // Threads per block
#define BLOCKS (N + THREADS - 1) / THREADS  // Number of blocks

// Naive Reduction (with bank conflicts)
__global__ void reduce_naive(float *input, float *output) {
    __shared__ float shared_mem[THREADS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load global memory into shared memory
    shared_mem[tid] = (global_idx < N) ? input[global_idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Write block sum to output
    if (tid == 0) output[blockIdx.x] = shared_mem[0];
}

// Optimized Reduction (Avoiding Bank Conflicts)
__global__ void reduce_optimized(float *input, float *output) {
    __shared__ float shared_mem[THREADS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory (coalesced)
    shared_mem[tid] = (global_idx < N) ? input[global_idx] : 0.0f;
    __syncthreads();

    // Reduction (Avoiding Bank Conflicts)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            shared_mem[index] += shared_mem[index + stride];
        }
        __syncthreads();
    }

    // Write block sum to output
    if (tid == 0) output[blockIdx.x] = shared_mem[0];
}

// CPU Reduction (Baseline for Benchmark)
float reduce_cpu(float *input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum;
}

// Benchmarking Function
void benchmark() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    cudaCheckError(::cudaMallocHost(&h_input, N * sizeof(float)));
    cudaCheckError(::cudaMallocHost(&h_output, BLOCKS * sizeof(float)));
    cudaCheckError(::cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheckError(::cudaMalloc(&d_output, BLOCKS * sizeof(float)));

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Simple case where sum = N
    }

    cudaCheckError(::cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaCheckError(::cudaEventCreate((&start)));
    cudaCheckError(::cudaEventCreate(&stop));
    // Benchmark Naïve Kernel
    cudaCheckError(::cudaEventRecord(start));
    reduce_naive<<<BLOCKS, THREADS>>>(d_input, d_output);
    cudaCheckError(::cudaEventRecord(stop));
    cudaCheckError(::cudaEventSynchronize(stop));
    float time_naive;
    cudaCheckError(::cudaEventElapsedTime(&time_naive, start, stop));
    printf("Naïve Kernel Time: %f ms\n", time_naive);
    cudaCheckError(::cudaMemcpy(h_output, d_output, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));

    // Final sum on CPU
    float sum_naive = reduce_cpu(h_output, BLOCKS);

    // Benchmark Optimized Kernel
    cudaCheckError(::cudaEventRecord(start));
    reduce_optimized<<<BLOCKS, THREADS>>>(d_input, d_output);
    cudaCheckError(::cudaEventRecord(stop));
    cudaCheckError(::cudaEventSynchronize(stop));
    float time_optimized;
    cudaCheckError(::cudaEventElapsedTime(&time_optimized, start, stop));
    printf("Optimized Kernel Time: %f ms\n", time_optimized);
    cudaCheckError(::cudaMemcpy(h_output, d_output, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum_optimized = reduce_cpu(h_output, BLOCKS);

    // CPU Baseline
    float sum_cpu = reduce_cpu(h_input, N);

    printf("CPU Sum: %f\n", sum_cpu);
    printf("Naïve GPU Sum: %f\n", sum_naive);
    printf("Optimized GPU Sum: %f\n", sum_optimized);

    // Cleanup
    cudaCheckError(::cudaFreeHost(h_input));
    cudaCheckError(::cudaFreeHost(h_output));
    cudaCheckError(::cudaFree(d_input));
    cudaCheckError(::cudaFree(d_output));
}

int main() {
    benchmark();
    return 0;
}
