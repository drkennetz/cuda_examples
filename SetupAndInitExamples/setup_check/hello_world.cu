#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../utils/utils.cuh"

// Device (GPU) code to print thread index from the GPU.
__global__ void helloFromGPU()
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread: %d!\n", threadId);
}

// Host code
int main()
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  major-minor: %d-%d\n", prop.major, prop.minor);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
    printf("  Managed memory supported: %s\n", prop.managedMemory ? "yes" : "no");
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes): %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes): %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  L2 cache size (Kbytes): %.1f\n",(float)(prop.l2CacheSize)/1024);
    printf("  Warp-size: %d\n", prop.warpSize);
  }
  printf("\n");
  printf("Hello from CPU!\n");
  helloFromGPU<<<1,4>>>();
  cudaCheckError(::cudaDeviceSynchronize());
  return 0;
}

