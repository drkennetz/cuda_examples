#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#include "../../utils/utils.cuh"

//#define M 1024  // Rows of A, Rows of C
//#define N 2048  // Columns of B, Columns of C
//#define K 512   // Columns of A, Rows of B

#define M 4096  // Rows of A, Rows of C
#define N 8192  // Columns of B, Columns of C
#define K 1024  // Columns of A, Rows of B

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
        //C[row * n + col] = value;  // Fixed: Use full matrix width 'n' as stride
        C[row * col_size + local_col] = value;
    }
}

// Host function for matrix multiplication
void matMulHost(const int* A, const int* B, int* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    // Host matrices
    int *A = new int[M * K];
    int *B = new int[K * N];
    int *C = new int[M * N]();  // GPU result
    int *C_host = new int[M * N]();  // CPU result

    // Initialize matrices with sample values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = (i + j) % 10;  // Simple pattern to avoid large numbers in multiplication
        }
    }
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = (i + j) % 10;
        }
    }

    // Calculate split for tensor parallelism across N dimension
    int cols_per_gpu = N / 2;
    
    // Arrays for each GPU
    int *d_A[2], *d_B[2], *d_C[2];
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start GPU timing
    cudaEventRecord(start);
    
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
        dim3 threadsPerBlock(16, 16);  // Using 16x16 thread blocks for better occupancy
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
                &C[row * N + gpu * cols_per_gpu],    // Destination in host matrix
                &d_C[gpu][row * cols_per_gpu],       // Source from GPU
                cols_per_gpu * sizeof(int),          // Size of this GPU's portion
                cudaMemcpyDeviceToHost
            ));
        }
    }

    // Stop GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Start CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // Compute result on CPU for verification
    std::cout << "Computing on CPU for verification..." << std::endl;
    matMulHost(A, B, C_host, M, N, K);
    
    // Stop CPU timing
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    // Print timing results
    std::cout << "\nTiming Results:" << std::endl;
    std::cout << "GPU Time: " << gpu_milliseconds << " ms" << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<float>(cpu_duration.count()) / gpu_milliseconds << "x" << std::endl;
    
    // Compare results
    bool match = true;
    int mismatch_count = 0;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != C_host[i]) {
            match = false;
            mismatch_count++;
        }
    }
    
    // Output comparison results
    std::cout << "\nResult comparison:" << std::endl;
    if (match) {
        std::cout << "GPU and CPU results match perfectly!" << std::endl;
    } else {
        std::cout << "Found " << mismatch_count << " mismatches between GPU and CPU results." << std::endl;
    }
    
    // Function to print a 4x4 region of both matrices
    auto print_region = [&](const char* region_name, int start_row, int start_col) {
        std::cout << "\n" << region_name << " (4x4):" << std::endl;
        std::cout << "GPU Result:" << std::endl;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << C[(start_row + i) * N + (start_col + j)] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "CPU Result:" << std::endl;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << C_host[(start_row + i) * N + (start_col + j)] << " ";
            }
            std::cout << std::endl;
        }
    };
    
    // Print different regions of the matrices
    print_region("Top-left corner", 0, 0);
    print_region("Middle region", M/2 - 2, N/2 - 2);
    print_region("Bottom-right corner", M - 4, N - 4);
    
    // Free device memory on each GPU
    for (int gpu = 0; gpu < 2; gpu++) {
        cudaCheckError(cudaSetDevice(gpu));
        cudaCheckError(cudaFree(d_A[gpu]));
        cudaCheckError(cudaFree(d_B[gpu]));
        cudaCheckError(cudaFree(d_C[gpu]));
    }

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_host;
    
    return 0;
}
