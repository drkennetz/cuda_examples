#include <iostream>
#include <cassert>
#include "../../utils/utils.cuh"

/**
 * @file vector_add.cu
 * 
 * @brief We demonstrate adding 2 vectors of size 2048 using CUDA
 */

/**
 * @brief CUDA kernel to perform vector addition
 * 
 * @param d_v1 Device vector holding a input vector
 * @param d_v2 Device vector holding another input vector
 * @param d_v3 Device vector holding the sum of d_v1 and d_v2
 */
__global__ void vector_add_kernel(float *d_v1, float *d_v2, float *d_v3, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        d_v3[i] = d_v1[i] + d_v2[i];
    }
}

int main() {
    int N = 2048;
    size_t size = sizeof(float) * N;

    // Allocate memory for the host vectors
    float *v1, *v2, *v3;
    cudaCheckError(::cudaMallocHost(&v1, size));
    cudaCheckError(::cudaMallocHost(&v2, size));
    cudaCheckError(::cudaMallocHost(&v3, size));

    // Fill the host vectors with data
    for (size_t i = 0; i < N; i++) {
        v1[i] = 6.4;
        v2[i] = 3.5;
    }

    // Create 3 device vectors for holding N float elements
    float *d_v1, *d_v2, *d_v3;
    cudaCheckError(::cudaMalloc(&d_v1, size));
    cudaCheckError(::cudaMalloc(&d_v2, size));
    cudaCheckError(::cudaMalloc(&d_v3, size));

    // Copy the host vectors into the device vectors
    cudaCheckError(::cudaMemcpy(d_v1, v1, size, cudaMemcpyHostToDevice));
    cudaCheckError(::cudaMemcpy(d_v2, v2, size, cudaMemcpyHostToDevice));

    // Launch the kernel using 8 thread blocks and 256 threads / block
    vector_add_kernel<<<8, 256>>>(d_v1, d_v2, d_v3, N);

    // Copy the result device vector into the result host vector
    cudaCheckError(::cudaMemcpy(v3, d_v3, size, cudaMemcpyDeviceToHost));

    cudaCheckError(::cudaDeviceSynchronize());

    // Verify the result host vector
    for (int i = 0; i < N; i++) {
        assert(v3[i] == (float)9.9);
    }

    // Free the device vector
    cudaCheckError(::cudaFree(d_v1));
    cudaCheckError(::cudaFree(d_v2));
    cudaCheckError(::cudaFree(d_v3));

    // Free the host vectors
    cudaCheckError(::cudaFreeHost(v1));
    cudaCheckError(::cudaFreeHost(v2));
    cudaCheckError(::cudaFreeHost(v3));

    return 0;
}
