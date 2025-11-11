# NCCL vs NIXL: KV Cache Transfer Benchmark

This example compares NCCL and NIXL for KV cache transfers in LLM inference workloads, replicating concepts from the [UCCL KV Transfer Engine article](https://uccl-project.github.io/posts/kv-transfer-engine/).

## Overview

In large language model (LLM) inference, especially with techniques like disaggregated inference or multi-GPU serving, efficiently transferring KV (key-value) caches between GPUs is critical for performance. This benchmark compares three approaches:

1. **Baseline cudaMemcpy**: Direct GPU-to-GPU memory copy
2. **NCCL (NVIDIA Collective Communication Library)**: Industry-standard collective communication library
3. **NIXL (NVIDIA Inference Transfer Library)**: Purpose-built for inference workloads

## Key Differences

### NCCL
- **Use case**: Primarily designed for training workloads
- **P2P transfers**: Requires launching GPU kernels
- **Resource usage**: Consumes GPU Streaming Multiprocessor (SM) resources
- **Impact**: Can interfere with concurrent inference computation
- **Best for**: Collective operations (all-reduce, all-gather) in training

### NIXL
- **Use case**: Optimized for inference workloads
- **P2P transfers**: Offloaded from GPU, no kernel launches needed
- **Resource usage**: Does NOT consume GPU SM resources
- **Impact**: Allows full GPU utilization for inference computation
- **Best for**: High-throughput, low-latency data transfers in inference

### Performance Characteristics

According to the UCCL project benchmarks:
- For typical KV cache message sizes (10s to 100s of MB), NIXL and UCCL P2P achieve **better performance** than NCCL
- NIXL provides **lower latency** due to not requiring GPU kernel launches
- NIXL enables **better overlap** of communication and computation

## Building

### Prerequisites

- CUDA Toolkit (12.x or later)
- C++20 compatible compiler
- Multi-GPU system (minimum 2 GPUs)

Optional:
- NCCL library (for NCCL comparison)
- NIXL library (for NIXL comparison)

### Build Options

#### 1. Baseline Only (cudaMemcpy)
```bash
make baseline
# or simply
make
```

#### 2. With NCCL Support
First, ensure NCCL is installed. Then edit the Makefile to set `NCCL_HOME` and uncomment the NCCL-related lines:
```bash
# Edit Makefile to set NCCL paths
make nccl
```

#### 3. With NIXL Support
First, build and install NIXL from [https://github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl). Then edit the Makefile:
```bash
# Edit Makefile to set NIXL paths
make nixl
```

#### 4. Full Comparison (Both NCCL and NIXL)
```bash
# Edit Makefile to set both NCCL and NIXL paths
make full
```

### CUDA Architecture

The Makefile defaults to `sm_86` (Ampere/RTX 30xx). Adjust for your GPU:
- `sm_70`: Volta (V100)
- `sm_75`: Turing (T4)
- `sm_80`: Ampere (A100)
- `sm_86`: Ampere (RTX 3090/A40)
- `sm_89`: Ada Lovelace (RTX 40xx)
- `sm_90`: Hopper (H100)

Edit the `CUDA_ARCH` variable in the Makefile.

## Running

```bash
./main
```

The benchmark will:
1. Detect available GPUs
2. Configure KV cache based on typical LLM parameters (LLaMA-2 70B by default)
3. Run warm-up iterations
4. Benchmark each method (cudaMemcpy, NCCL, NIXL) with 100 iterations
5. Report latency and throughput statistics

### Expected Output

```
============================================
NCCL vs NIXL: KV Cache Transfer Benchmark
============================================

Detected 2 GPU(s)
  GPU 0: NVIDIA A100-SXM4-40GB (Compute 8.0)
  GPU 1: NVIDIA A100-SXM4-40GB (Compute 8.0)

KV Cache Configuration:
  Layers:      80
  Heads:       64
  Head Dim:    128
  Seq Length:  2048
  Batch Size:  1
  Cache Size:  5120 MB per GPU

Running 100 iterations per method...

=== Baseline cudaMemcpy Benchmark ===
...
```

## KV Cache Configuration

The benchmark simulates a typical LLaMA-2 70B model configuration:
- 80 transformer layers
- 64 attention heads per layer
- 128 dimensions per head
- 2048 token sequence length
- Batch size of 1 (inference)

This results in ~5GB of KV cache per GPU, typical for large model inference.

## Installing NIXL

To use NIXL, you need to build it from source. See the [NIXL GitHub repository](https://github.com/ai-dynamo/nixl) for full installation instructions.

Quick start:
```bash
# Install dependencies
sudo apt install build-essential cmake pkg-config
pip3 install meson ninja pybind11 tomlkit

# Build UCX (required by NIXL)
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.20.x
./autogen.sh
./configure --with-cuda=/usr/local/cuda --enable-mt
make -j
sudo make install

# Build NIXL
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
meson setup build
cd build
ninja
sudo ninja install
```

## References

- [UCCL KV Transfer Engine](https://uccl-project.github.io/posts/kv-transfer-engine/)
- [NIXL GitHub](https://github.com/ai-dynamo/nixl)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [UCX Communication Framework](https://openucx.org/)

## Performance Notes

### Why NIXL is Better for Inference

1. **No SM Resource Consumption**: NCCL's P2P operations launch GPU kernels that consume SMs, reducing the compute available for inference. NIXL offloads transfers entirely.

2. **Lower Latency**: By avoiding kernel launch overhead, NIXL achieves lower latency for the message sizes typical in KV cache transfers (tens to hundreds of MB).

3. **Better Overlap**: NIXL enables true overlap of communication and computation, maximizing GPU utilization during inference.

4. **Purpose-Built**: While NCCL excels at collective operations for training, NIXL is designed specifically for inference data movement patterns.

## Limitations

- Requires minimum 2 GPUs for meaningful comparison
- NCCL and NIXL must be installed separately (not included in CUDA Toolkit)
- Performance varies based on GPU generation, NVLink topology, and system configuration
- This is a micro-benchmark; real-world inference systems have additional complexity

## Further Exploration

To extend this example:
1. Test with different KV cache sizes (vary sequence length, model size)
2. Measure impact on concurrent computation (launch compute kernels alongside transfers)
3. Benchmark with more GPUs (4, 8, etc.) in various topologies
4. Profile SM utilization to demonstrate NCCL's resource consumption
5. Test with disaggregated inference patterns (prefix caching, dynamic batching)

## Contributing

Improvements welcome! Some ideas:
- Add actual NIXL API integration (currently stubbed)
- Add NCCL group call optimizations
- Add profiling with NSight Systems
- Add tests for different network topologies (NVLink, PCIe, InfiniBand)
- Add comparison with other libraries (UCX directly, MPI, etc.)

