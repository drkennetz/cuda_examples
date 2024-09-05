#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include "../../utils/utils.cuh"

// Simple matmul kernel.
__global__ void matMul(float *out, const float *in1, const float *in2, const int width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < width && col < width) {
        for (int i = 0; i < width; ++i) {
            sum += in1[row * width + i] * in2[i * width + col];
        }
        out[row * width + col] = sum;
    }
}

// Add one kernel to emphasize serial execution within a stream.
__global__ void addOne(float *out, const int width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        int idx = row * width + col;
        out[idx] += 1.0f;
    }
}

// Minus one kernel to emphasize serial execution within a stream.
__global__ void minusOne(float *out, const int width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        int idx = row * width + col;
        out[idx] -= 1.0f;
    }
}

// Add matrices to a final result for comparison.
__global__ void matAdd(float *out, const float *in1, const float *in2, const int width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        out[row * width + col] = in1[row * width + col] + in2[row * width + col];
    }
}

// Perform a matmul on separate streams for a million elements, then add them.
int main() {
    // Set dimensionality.
    const int width = 1024;
    const int size = width * width * sizeof(float);

    // Create separate streams to run matmuls asynchronously.
    cudaStream_t stream1, stream2;
    cudaCheckError(::cudaStreamCreate(&stream1));
    cudaCheckError(::cudaStreamCreate(&stream2));

    // Allocations and initializations.
    float *hA, *hB, *hC, *hD, *hResult;
    float *dA, *dB, *dC, *dD, *dResultAB, *dResultCD, *dFinal;

    // Host.
    cudaCheckError(::cudaMallocHost(&hA, size));
    cudaCheckError(::cudaMallocHost(&hB, size));
    cudaCheckError(::cudaMallocHost(&hC, size));
    cudaCheckError(::cudaMallocHost(&hD, size));
    cudaCheckError(::cudaMallocHost(&hResult, size));

    for (int i = 0; i < width * width; ++i) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
        hC[i] = 3.0f;
        hD[i] = 4.0f;
    }

    // Device.
    cudaCheckError(::cudaMalloc(&dA, size));
    cudaCheckError(::cudaMalloc(&dB, size));
    cudaCheckError(::cudaMalloc(&dC, size));
    cudaCheckError(::cudaMalloc(&dD, size));
    cudaCheckError(::cudaMalloc(&dResultAB, size));
    cudaCheckError(::cudaMalloc(&dResultCD, size));
    cudaCheckError(::cudaMalloc(&dFinal, size));

    // Copy data using separate streams. This is not necessary, per se, but it will
    // allow for overlap of copy and execution in a batch mode case.
    cudaCheckError(::cudaMemcpyAsync(dA, hA, size, cudaMemcpyHostToDevice, stream1));
    cudaCheckError(::cudaMemcpyAsync(dB, hB, size, cudaMemcpyHostToDevice, stream1));
    cudaCheckError(::cudaMemcpyAsync(dC, hC, size, cudaMemcpyHostToDevice, stream2));
    cudaCheckError(::cudaMemcpyAsync(dD, hD, size, cudaMemcpyHostToDevice, stream2));
    
    // Launch matmul kernels on different streams.
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMul<<<numBlocks, threadsPerBlock, 0, stream1>>>(dResultAB, dA, dB, width);
    matMul<<<numBlocks, threadsPerBlock, 0, stream2>>>(dResultCD, dC, dD, width);

    // Add 1.0f to each element in the result matrices
    addOne<<<numBlocks, threadsPerBlock, 0, stream1>>>(dResultAB, width);
    addOne<<<numBlocks, threadsPerBlock, 0, stream2>>>(dResultCD, width);

    // Subtract 1.0f to each element in the result matrices
    minusOne<<<numBlocks, threadsPerBlock, 0, stream1>>>(dResultAB, width);
    minusOne<<<numBlocks, threadsPerBlock, 0, stream2>>>(dResultCD, width);
    
    // Synchronize work prior to computing on each stream's result array.
    cudaCheckError(::cudaStreamSynchronize(stream1));
    cudaCheckError(::cudaStreamSynchronize(stream2));

    matAdd<<<numBlocks, threadsPerBlock, 0, stream1>>>(dFinal, dResultAB, dResultCD, width);
    cudaCheckError(::cudaMemcpy(hResult, dFinal, size, cudaMemcpyDeviceToHost));

    // Verify results.
    float expected = (1.0f * 2.0f + 3.0f * 4.0f) * width;
    for (int i = 0; i < width * width; ++i) {
        assert(fabs(hResult[i] - expected) < 1e-5);
    }

    std::cout << "Test passed!" << std::endl;

    // Clean up.
    cudaCheckError(::cudaFree(dA));
    cudaCheckError(::cudaFree(dB));
    cudaCheckError(::cudaFree(dC));
    cudaCheckError(::cudaFree(dD));
    cudaCheckError(::cudaFree(dResultAB));
    cudaCheckError(::cudaFree(dResultCD));
    cudaCheckError(::cudaFree(dFinal));
    cudaCheckError(::cudaStreamDestroy(stream1));
    cudaCheckError(::cudaStreamDestroy(stream2));
    cudaCheckError(::cudaFreeHost(hA));
    cudaCheckError(::cudaFreeHost(hB));
    cudaCheckError(::cudaFreeHost(hC));
    cudaCheckError(::cudaFreeHost(hD));
    cudaCheckError(::cudaFreeHost(hResult));

    return 0;
}

