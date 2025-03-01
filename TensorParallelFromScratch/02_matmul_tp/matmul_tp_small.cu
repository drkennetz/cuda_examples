#include <iostream>
#include <cuda_runtime.h>

#include "../../utils/utils.cuh"

#define M 2  // Rows of A, Rows of C
#define N 2  // Columns of B, Columns of C
#define K 3  // Columns of A, Rows of B

// Kernel to perform matrix multiplication on a portion of the matrices
__global__ void matMulKernelTP(int *A, int *B, int *C, int m, int n, int k, int col_start, int col_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_start + local_col;
    
    if (row < m && local_col < col_size) {
        int value = 0;
        for (int i = 0; i < k; i++) {
            value += A[row * k + i] * B[i * n + col];
        }
        C[row * col_size + local_col] = value;
    }
}

int main() {
    // Host matrices
    int A[M][K] = { {1, 2, 3}, {4, 5, 6} };
    int B[K][N] = { {100, 200}, {300, 400}, {500, 600} };
    int C[M][N] = {0};

    // Calculate split for tensor parallelism across N dimension
    int cols_per_gpu = N / 2;
    
    // Arrays for each GPU
    int *d_A[2], *d_B[2], *d_C[2];
    
    // Set up each GPU
    for (int gpu = 0; gpu < 2; gpu++) {
        cudaCheckError(cudaSetDevice(gpu));
        
        // Allocate memory on current GPU
        cudaCheckError(cudaMalloc(&d_A[gpu], M * K * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_B[gpu], K * N * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_C[gpu], M * cols_per_gpu * sizeof(int)));
        
        // Copy input matrices to current GPU
        cudaCheckError(cudaMemcpy(d_A[gpu], A, M * K * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_B[gpu], B, K * N * sizeof(int), cudaMemcpyHostToDevice));
        
        // Set grid and block dimensions for this GPU's portion
        dim3 threadsPerBlock(cols_per_gpu, M);
        dim3 numBlocks(
            (cols_per_gpu + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (M + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        
        // Launch kernel for this GPU's portion
        int col_start = gpu * cols_per_gpu;
        matMulKernelTP<<<numBlocks, threadsPerBlock>>>(
            d_A[gpu], d_B[gpu], d_C[gpu],
            M, N, K, col_start, cols_per_gpu
        );
    }
    
    // Copy results back from each GPU and combine
    for (int gpu = 0; gpu < 2; gpu++) {
        cudaCheckError(cudaSetDevice(gpu));
        // Copy each row's portion separately to maintain correct layout
        for (int row = 0; row < M; row++) {
            cudaCheckError(cudaMemcpy(
                &C[row][gpu * cols_per_gpu],         // Destination in host matrix
                &d_C[gpu][row * cols_per_gpu],       // Source from GPU
                cols_per_gpu * sizeof(int),          // Size of this GPU's portion
                cudaMemcpyDeviceToHost
            ));
        }
    }
    
    // Output the result matrix
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Free device memory on each GPU
    for (int gpu = 0; gpu < 2; gpu++) {
        cudaCheckError(cudaSetDevice(gpu));
        cudaCheckError(cudaFree(d_A[gpu]));
        cudaCheckError(cudaFree(d_B[gpu]));
        cudaCheckError(cudaFree(d_C[gpu]));
    }
    
    return 0;
}
