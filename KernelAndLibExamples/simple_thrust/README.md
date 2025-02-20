# CUDA Thrust Example: Parallel Element Transformation

## Overview

This project demonstrates the use of **Thrust**, a high-level parallel programming library for CUDA, to efficiently transform elements in a large vector using the `thrust::for_each` function. The program initializes a **32-million-element host vector**, transfers it to the GPU, applies a transformation using Thrust, and copies the modified data back to the host.

## Implementation Details

### **Key Components**
1. **Host Vector Initialization**  
   - The host vector (`thrust::host_vector<uint8>`) is allocated with **32 million elements**.
   - Each element is assigned a value based on `i & 0xFF`, effectively storing only the lower 8 bits of the index.

2. **Device Vector & Transformation**  
   - The vector is copied to a `thrust::device_vector<uint8>`.
   - The transformation is applied using `thrust::for_each` with a **lambda function** executed on the GPU.
   - The transformation follows this rule:
     - If the value is **255**, it is set to **0**.
     - Otherwise, it is set to **1**.

3. **Data Transfer & Verification**  
   - The modified vector is copied back to the host.
   - The first 10 elements of the vector are printed **before and after GPU processing** to verify the transformation.

---

## **Building and Running the Program**

### **Prerequisites**
- **CUDA 12.3+**
- **NVIDIA GPU with Compute Capability 8.6+ (e.g., A100, H100)**
- **NVIDIA CUDA Compiler (`nvcc`)**
- **Thrust (included in CUDA by default)**

### **Build Instructions**
Use the provided `Makefile` to compile the program:

```sh
make
```

This compiles simple_thrust.cu into an executable named main using nvcc with:

 - -std=c++20: Enables modern C++ support.
 - --extended-lambda: Allows GPU lambda expressions in Thrust.
 - -arch=sm_86: Targets NVIDIA Ampere GPUs.

Run the Program
Execute the compiled binary:
```bash
./main
```
## Expected output
You should see output similar to:
```bash
hDataSize: 32000000
Host data pre GPU for_each:
0 1 2 3 4 5 6 7 8 9
Host data post GPU for_each:
1 1 1 1 1 1 1 1 1 1
```
This confirms that:
 - The input vector caontained values from 0-9 initially.
 - After GPU processing, all elements except 255 were set to 1.

## Key Thrust APIs Used
 - `thrust::host_vector<T>`: Manages host-side vectors.
 - `thrust::device_vector<T>`: Allocates and manages GPU memory.
 - `thrust::for_each(begin, end, functor)`: Applies a parallel operation to each element

## Why Use Thrust?
 - Simplifies CUDA programming: No need to manually allocate/deallocate device memory.
 - Optimized parallel execution: Efficient kernel launches under the hood.
 - Readable & expressive code: Replaces explicit CUDA kernels with high-level abstractions.

## When Not to Use Thrust

While Thrust is powerful, there are situations where it may not be the best choice:

 - Custom kernel optimizations: If you need fine-tuned memory access patterns, shared memory usage, or warp-level optimizations, writing a raw CUDA kernel may provide better performance.
 - Complex control flow: If your workload involves deep branching, recursion, or irregular memory access, Thrust may not be as flexible as custom CUDA kernels.
 - Small data sizes: Thrust is optimized for large datasets. For small workloads, the overhead of launching Thrust kernels may outweigh the benefits.
 - Specialized operations: If you need Tensor Core optimizations, mixed-precision arithmetic, or other low-level CUDA-specific features, Thrust may not provide direct support.