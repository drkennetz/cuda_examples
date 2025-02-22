#include <iostream>
#include <cuda_runtime.h>

#include "../../utils/utils.cuh"

#define M 2  // Rows of A, Rows of C
#define N 2  // Columns of B, Columns of C
#define K 3  // Columns of A, Rows of B

// Kernel to perform our simple matrix multiplication
__global__ void matMulKernel(int *A, int *B, int *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("row: %d, col: %d\n", row, col);
    if (row < m && col < n) {
        int value = 0;
        for (int i = 0; i < k; i++) {
            value += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = value;
    }
}

int main() {
    // Host matrices
    int A[M][K] = { {1, 2, 3}, {4, 5, 6} };
    int B[K][N] = { {100, 200}, {300, 400}, {500, 600} };
    int C[N][M] = {0};

    int *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaCheckError(::cudaMalloc(&d_A, M * K * sizeof(int)));
    cudaCheckError(::cudaMalloc(&d_B, K * N * sizeof(int)));
    cudaCheckError(::cudaMalloc(&d_C, M * N * sizeof(int)));

    // Copy matrices from host to device
    cudaCheckError(::cudaMemcpy(d_A, A, M * K * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(::cudaMemcpy(d_B, B, K * N * sizeof(int), cudaMemcpyHostToDevice));

    // Set our grid and block size
    dim3 threadsPerBlock(M, N);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaCheckError(::cudaMemcpy(C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost));

    // Output the result matrix
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaCheckError(::cudaFree(d_A));
    cudaCheckError(::cudaFree(d_B));
    cudaCheckError(::cudaFree(d_C));

    return 0;
}
