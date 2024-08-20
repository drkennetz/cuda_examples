#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <iostream>

// Kernel to intialize weights with random values.
__global__ void initializeWeightsKernel(float* weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandStateXORWOW state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state);
    }
}

// Function to initialize weights using cudaMalloc
void initializeWeightsCudaMalloc(int size) {

    float* d_weights;
    // Allocate memory for distribution, but don't initialize values until kernel
    cudaMalloc(&d_weights, size * sizeof(float));

    // Launch kernel to initialize weights
    initializeWeightsKernel<<<(size + 255) / 256, 256>>>(d_weights, size, time(NULL));
    cudaDeviceSynchronize();

    cudaFree(d_weights);

    std::cout << "Initialized weights using cudaMalloc." << std::endl;
}

// Function to initialize weights using thrust::device_vector
void initializeWeightsThrust(int size) {

    // Create a device vector and initialize with zeros
    thrust::device_vector<float> weights(size);

    // Launch kernel to initialize weights
    initializeWeightsKernel<<<(size + 255) / 256, 256>>>(thrust::raw_pointer_cast(weights.data()), size, time(NULL));

    std::cout << "Initialized weights using thrust::device_vector." << std::endl;
}

int main() {
    const int size = 1000000;  // Large number of weights

    // Initialize weights using thrust::device_vector
    initializeWeightsThrust(size);

    // Initialize weights using cudaMalloc
    initializeWeightsCudaMalloc(size);

    return 0;
}
