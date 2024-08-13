#include <cstdio>
#include <cuda_runtime.h>
constexpr int NUM_COPY_ITERATIONS = 1000;
void checkBandwidth(cudaStream_t& stream, size_t dataSize) {
    // Allocate and load host memory.
    float* hData = new float[dataSize];
    for (size_t i = 0; i < dataSize; ++i) {
        hData[i] = static_cast<float>(i);
    }
    // Allocate device memory.
    float* dData;
    ::cudaMalloc(&dData, dataSize * sizeof(float));
    ::cudaStreamSynchronize(stream);
    // Create CUDA events for timing purposes.
    cudaEvent_t start, stop;
    ::cudaEventCreate(&start);
    ::cudaEventCreate(&stop);
    // Record the start event.
    ::cudaEventRecord(start, stream);
    for (int i = 0; i < NUM_COPY_ITERATIONS; ++i) {
        ::cudaMemcpyAsync(dData, hData, dataSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    // Record the stop event.
    ::cudaEventRecord(stop, stream);
    // Sync the stream to ensure copy complete.
    ::cudaStreamSynchronize(stream);
    // Calculate the elapsed time.
    float ms = 0;
    ::cudaEventElapsedTime(&ms, start, stop);
    // Convert to GB/s.
    const float bandwidth = ((float(dataSize) * sizeof(float) * float(NUM_COPY_ITERATIONS)) / (ms * 1e6));
    printf("Bandwidth: %f GB/s\n", bandwidth);
    // Clean up.
    ::cudaFree(dData);
    delete[] hData;
    ::cudaEventDestroy(start);
    ::cudaEventDestroy(stop);
}

int main() {
    cudaStream_t stream;
    ::cudaStreamCreate(&stream);

    const size_t dataSize = 1 << 20;
    checkBandwidth(stream, dataSize);
    ::cudaStreamDestroy(stream);
    return 0;
}
