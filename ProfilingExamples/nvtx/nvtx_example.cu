#include <cstdio>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include "../../utils/utils.cuh"

#define W 1024

__global__ void matMul(float* out, const float* in1, const float* in2, const int width) {
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

void doMatMul(float* out, const float* in1, const float* in2, int width) {
    float *dIn1, *dIn2, *dOut;
    const size_t size = width * width * sizeof(float);

    printf("Allocating device memory...\n");
    cudaCheckError(::cudaMalloc(&dIn1, size));
    cudaCheckError(::cudaMalloc(&dIn2, size));
    cudaCheckError(::cudaMalloc(&dOut, size));

    printf("Copying data to device...\n");
    cudaCheckError(::cudaMemcpy(dIn1, in1, size, cudaMemcpyHostToDevice));
    cudaCheckError(::cudaMemcpy(dIn2, in2, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    printf("Launching kernel...\n");
    for (int j = 0; j < 100; ++j)
    {
        nvtxRangePush("Matrix multiplication kernel");
        matMul<<<blocksPerGrid, threadsPerBlock>>>(dOut, dIn1, dIn2, width);
        nvtxRangePop();
    }
    cudaCheckError(::cudaDeviceSynchronize());

    printf("Copying result to host...\n");
    cudaCheckError(::cudaMemcpy(out, dOut, size, cudaMemcpyHostToDevice));

    printf("Freeing device memory...\n");
    cudaCheckError(::cudaFree(dIn1));
    cudaCheckError(::cudaFree(dIn2));
    cudaCheckError(::cudaFree(dOut));
}

int main() {
    float *in1, *in2, *out;
    const size_t width = W * W * sizeof(float);
    cudaCheckError(::cudaMallocHost(&in1, width));
    cudaCheckError(::cudaMallocHost(&in2, width));
    cudaCheckError(::cudaMallocHost(&out, width));
    for (int i = 0; i < W * W; ++i) {
        in1[i] = static_cast<float>(rand()) / RAND_MAX;
        in2[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    printf("Starting matrix multiplication...\n");
    doMatMul(out, in1, in2, width);
    printf("Matrix multiplication completed.\n");
    printf("Freeing host memory...\n");
    cudaCheckError(::cudaFreeHost(in1));
    cudaCheckError(::cudaFreeHost(in2));
    cudaCheckError(::cudaFreeHost(out));
    return 0;
}
