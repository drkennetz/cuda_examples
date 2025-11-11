# Quick Start Guide

Get started with the NCCL vs NIXL benchmark in 5 minutes!

## TL;DR

```bash
# Navigate to the example
cd TensorParallelFromScratch/03_nccl_vs_nixl

# Run the baseline benchmark (works immediately)
./test_baseline.sh --run
```

## What This Example Does

Compares three methods for transferring KV cache data between GPUs in LLM inference:
1. **cudaMemcpy** (baseline) - Simple P2P memory copy
2. **NCCL** - NVIDIA's collective communication library (requires installation)
3. **NIXL** - NVIDIA's inference transfer library (requires installation)

## Quick Commands

### Run Baseline (No Installation Required)
```bash
make baseline
./main
```

### Check System Requirements
```bash
# Check for GPUs
nvidia-smi

# Check CUDA
nvcc --version

# Check for NCCL
ldconfig -p | grep nccl

# Check for NIXL
ldconfig -p | grep nixl
```

### Build Different Versions

```bash
# Baseline only
make baseline

# With NCCL (requires NCCL installed)
make nccl

# With NIXL (requires NIXL installed)
make nixl

# With both
make full
```

## Installation Summary

### NCCL (Recommended First Step)
```bash
# Ubuntu/Debian - easiest method
sudo apt-get update
sudo apt-get install libnccl2 libnccl-dev

# Then rebuild
make nccl
```

### NIXL (More Involved)
```bash
# Install dependencies
sudo apt-get install -y build-essential cmake pkg-config
pip3 install --user meson ninja pybind11 tomlkit

# Build UCX (required by NIXL)
git clone https://github.com/openucx/ucx.git && cd ucx && git checkout v1.20.0
./autogen.sh && ./configure --with-cuda=/usr/local/cuda --enable-mt
make -j$(nproc) && sudo make install && sudo ldconfig

# Build NIXL
git clone https://github.com/ai-dynamo/nixl.git && cd nixl
meson setup build && cd build && ninja && sudo ninja install && sudo ldconfig

# Then rebuild
make nixl
```

See `INSTALL.md` for detailed instructions.

## Understanding the Output

```
Detected 8 GPU(s)          ← Number of CUDA-capable GPUs found
Cache Size: 10240 MB       ← Total KV cache size per GPU
Average Latency: 40.5 ms   ← Time to transfer between all GPUs
Throughput: 1973.56 GB/s   ← Data transfer bandwidth
```

### What Good Results Look Like

**cudaMemcpy Baseline:**
- Latency: 30-50 ms (depends on GPU count and topology)
- Throughput: 200-400 GB/s per GPU pair with NVLink
- Throughput: 50-100 GB/s per GPU pair with PCIe

**NCCL:**
- Similar or slightly better than baseline
- BUT: Consumes GPU SM resources (bad for inference)
- Good for collective operations (all-reduce, etc.)

**NIXL:**
- Lower latency than NCCL (10-30% improvement typical)
- Zero SM resource consumption (key advantage!)
- Best for overlapping with computation

## Key Insights

### Why This Matters

In LLM inference with multiple GPUs:
1. KV caches must be shared between GPUs
2. Transfers can happen frequently (every token in some cases)
3. Any SM usage for transfers = less compute for inference
4. Lower latency = faster response times

### NCCL vs NIXL Decision Tree

**Use NCCL when:**
- Training workloads
- Need collective operations (all-reduce, all-gather)
- Already using NCCL for other operations
- SM resource contention is acceptable

**Use NIXL when:**
- Inference workloads
- P2P transfers are primary pattern
- Need maximum GPU utilization
- Latency is critical

**Use cudaMemcpy when:**
- Simple single-stream applications
- Prototyping
- Minimal dependencies required

## Configuration Options

Edit the source file `nixl_vs_nccl.cu` to change:

### KV Cache Size
```cpp
KVCacheConfig config;
config.num_layers = 80;      // LLaMA-2 70B has 80 layers
config.num_heads = 64;       // Number of attention heads
config.head_dim = 128;       // Dimension per head
config.seq_len = 2048;       // Sequence length
```

Try different model sizes:
- **LLaMA-2 7B**: 32 layers, 32 heads, 128 dim
- **LLaMA-2 13B**: 40 layers, 40 heads, 128 dim
- **LLaMA-2 70B**: 80 layers, 64 heads, 128 dim (default)
- **GPT-3 175B**: 96 layers, 96 heads, 128 dim

### Number of Iterations
```cpp
int iterations = 100;  // Line ~700 in main()
```

More iterations = more accurate statistics, longer runtime.

## Troubleshooting

### "No CUDA-capable device"
```bash
# Check driver
nvidia-smi

# If fails, driver issue - reinstall NVIDIA driver
```

### "CMAKE_CUDA_COMPILER could not be found"
```bash
# Set CUDA in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Compilation Errors
```bash
# Check CUDA architecture matches your GPU
# Edit Makefile, change CUDA_ARCH = -arch=sm_XX

# Common values:
# sm_70 = V100
# sm_80 = A100
# sm_86 = RTX 3090/A40
# sm_89 = RTX 4090
# sm_90 = H100
```

### Low Performance
```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Check NVLink status (if available)
nvidia-smi nvlink --status

# Check for peer access
nvidia-smi topo -m
```

## Next Steps

1. **Run baseline** to verify system works
2. **Install NCCL** (easy via apt) and compare
3. **Install NIXL** (more involved) for full comparison
4. **Experiment** with different model sizes
5. **Profile** with NSight Systems to see SM usage differences

## Example Session

```bash
ubuntu@system:~$ cd cuda_examples/TensorParallelFromScratch/03_nccl_vs_nixl
ubuntu@system:~/...03_nccl_vs_nixl$ ./test_baseline.sh --run

===========================================
NCCL vs NIXL Baseline Test
===========================================

Checking CUDA installation...
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.6, V12.6.85

Checking for GPUs...
Found 8 GPU(s)

Building baseline benchmark...
nvcc -arch=sm_86 -std=c++20 -O3 -I../../utils -o main nixl_vs_nccl.cu

Build successful!

Running benchmark...
===========================================
============================================
NCCL vs NIXL: KV Cache Transfer Benchmark
============================================

Detected 8 GPU(s)
...
Average Latency: 40.536 ms
Throughput: 1973.56 GB/s
============================================
```

## Learning Resources

- **UCCL Blog Post**: [https://uccl-project.github.io/posts/kv-transfer-engine/](https://uccl-project.github.io/posts/kv-transfer-engine/)
- **NCCL Docs**: [https://docs.nvidia.com/deeplearning/nccl/](https://docs.nvidia.com/deeplearning/nccl/)
- **NIXL GitHub**: [https://github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)
- **Detailed Install Guide**: See `INSTALL.md` in this directory
- **Implementation Details**: See `SUMMARY.md` in this directory

## Common Use Cases

### Benchmark Your System
```bash
make baseline
./main > results_baseline.txt

# After installing NCCL
make nccl
./main > results_nccl.txt

# After installing NIXL
make nixl
./main > results_nixl.txt

# Compare results
diff results_baseline.txt results_nccl.txt
```

### Test Different Model Sizes
```bash
# Edit nixl_vs_nccl.cu, change config parameters
# For example, test LLaMA-2 7B instead of 70B:
# config.num_layers = 32;
# config.num_heads = 32;

make baseline
./main
```

### Profile with NSight Systems
```bash
make nccl
nsys profile --stats=true ./main
```

## Getting Help

- **General questions**: Open issue in cuda_examples repository
- **NCCL issues**: [https://github.com/NVIDIA/nccl/issues](https://github.com/NVIDIA/nccl/issues)
- **NIXL issues**: [https://github.com/ai-dynamo/nixl/issues](https://github.com/ai-dynamo/nixl/issues)
- **CUDA issues**: NVIDIA Developer Forums

## FAQ

**Q: Why do I get "Insufficient GPUs" error?**  
A: The benchmark requires 2+ GPUs. The code can be modified to test on 1 GPU using streams (not representative of real performance).

**Q: Can I run this without NCCL/NIXL?**  
A: Yes! The baseline (cudaMemcpy) works without any additional libraries.

**Q: What's the best GPU topology?**  
A: NVLink/NVSwitch provides best performance. PCIe works but is slower.

**Q: Should I use this in production?**  
A: This is an educational example. Production systems need error handling, monitoring, resource management, etc.

**Q: How do I add more GPUs?**  
A: The code automatically detects and uses all available GPUs.

**Q: Why is NIXL better for inference?**  
A: NIXL doesn't consume GPU SM resources, leaving full GPU capacity for inference computation.

---

**Ready to start?** Run:
```bash
./test_baseline.sh --run
```

