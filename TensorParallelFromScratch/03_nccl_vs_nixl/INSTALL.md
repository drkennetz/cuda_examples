# Installation Guide for NCCL and NIXL

This guide provides step-by-step instructions for installing NCCL and NIXL to run the full comparison benchmark.

## Prerequisites

- Ubuntu 20.04+ (or similar Linux distribution)
- CUDA Toolkit 12.x (already installed at `/usr/local/cuda`)
- Build tools: `build-essential`, `cmake`, `pkg-config`
- Python 3.8+ with pip
- Multi-GPU system (minimum 2 GPUs)

## Setting Up CUDA PATH (Required)

Before building anything, ensure CUDA is in your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

To make this permanent, add to your `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify CUDA is accessible:

```bash
nvcc --version
nvidia-smi
```

## Installing NCCL

NCCL (NVIDIA Collective Communication Library) is available through several methods:

### Option 1: Install from NVIDIA Repository (Recommended)

```bash
# Add NVIDIA repository (if not already added)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install NCCL
sudo apt-get install libnccl2 libnccl-dev

# Verify installation
dpkg -l | grep nccl
```

### Option 2: Build from Source

```bash
# Clone NCCL repository
git clone https://github.com/NVIDIA/nccl.git
cd nccl

# Checkout latest stable version
git checkout v2.21.5-1  # or latest version

# Build
make -j src.build

# Install
sudo make install

# Add to library path
echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf.d/nccl.conf
sudo ldconfig
```

### Verify NCCL Installation

```bash
# Check library
ldconfig -p | grep nccl

# Check headers
ls /usr/include/nccl.h || ls /usr/local/include/nccl.h
```

## Installing NIXL

NIXL (NVIDIA Inference Transfer Library) requires UCX as a dependency and must be built from source.

### Step 1: Install Build Dependencies

```bash
# Install required packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    autoconf \
    automake \
    libtool \
    flex \
    librdmacm-dev \
    libibverbs-dev \
    libnuma-dev

# Install Python build tools
pip3 install --user meson ninja pybind11 tomlkit
```

### Step 2: Install UCX (Required Dependency)

```bash
# Clone UCX
cd /tmp
git clone https://github.com/openucx/ucx.git
cd ucx

# Checkout stable version
git checkout v1.20.0  # or latest v1.20.x

# Generate configure script
./autogen.sh

# Configure with CUDA support
./configure \
    --prefix=/usr/local \
    --enable-shared \
    --disable-static \
    --disable-doxygen-doc \
    --enable-optimizations \
    --enable-cma \
    --enable-devel-headers \
    --with-cuda=/usr/local/cuda \
    --with-verbs \
    --with-dm \
    --enable-mt

# Build (this may take 10-15 minutes)
make -j$(nproc)

# Install
sudo make install

# Update library cache
sudo ldconfig

# Verify UCX installation
ucx_info -v
```

### Step 3: Optional - Install GDRCopy (For Better Performance)

GDRCopy enables GPUDirect RDMA for maximum performance:

```bash
# Clone GDRCopy
cd /tmp
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy

# Build
make -j$(nproc)

# Install
sudo make install

# Load kernel module
sudo modprobe gdrdrv

# Make module load at boot
echo "gdrdrv" | sudo tee -a /etc/modules
```

Then rebuild UCX with GDRCopy support:

```bash
cd /tmp/ucx
make clean
./configure \
    --prefix=/usr/local \
    --enable-shared \
    --disable-static \
    --enable-optimizations \
    --with-cuda=/usr/local/cuda \
    --with-verbs \
    --with-gdrcopy=/usr/local \
    --enable-mt
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Step 4: Install NIXL

```bash
# Clone NIXL
cd /tmp
git clone https://github.com/ai-dynamo/nixl.git
cd nixl

# Setup build with meson
meson setup build

# Build (this may take 5-10 minutes)
cd build
ninja

# Install
sudo ninja install

# Update library cache
sudo ldconfig

# Verify installation
ls /usr/local/include/nixl.h
ldconfig -p | grep nixl
```

## Building the Benchmark

Once NCCL and NIXL are installed, update the Makefile in this directory:

### Edit Makefile

Open `Makefile` and uncomment/update the following sections:

```makefile
# For NCCL
NCCL_HOME = /usr/local
NCCL_INCLUDE = -I$(NCCL_HOME)/include
NCCL_LIB = -L$(NCCL_HOME)/lib -lnccl
NCCL_FLAGS = -DUSE_NCCL
INCLUDES += $(NCCL_INCLUDE)
LIBS += $(NCCL_LIB)
NVCCFLAGS += $(NCCL_FLAGS)

# For NIXL
NIXL_HOME = /usr/local
NIXL_INCLUDE = -I$(NIXL_HOME)/include
NIXL_LIB = -L$(NIXL_HOME)/lib -lnixl
NIXL_FLAGS = -DUSE_NIXL
INCLUDES += $(NIXL_INCLUDE)
LIBS += $(NIXL_LIB)
NVCCFLAGS += $(NIXL_FLAGS)
```

### Build Options

```bash
# Build with both NCCL and NIXL
make full

# Or build with just NCCL
make nccl

# Or build with just NIXL
make nixl

# Or build baseline only (cudaMemcpy)
make baseline
```

## Troubleshooting

### NCCL Issues

**Problem**: `libnccl.so` not found

```bash
# Find NCCL library
find /usr -name "libnccl.so*" 2>/dev/null

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH
```

**Problem**: NCCL version mismatch

```bash
# Check NCCL version
dpkg -l | grep nccl
# or
strings /usr/lib/x86_64-linux-gnu/libnccl.so.2 | grep NCCL_VERSION
```

### NIXL/UCX Issues

**Problem**: `libucx.so` or `libnixl.so` not found

```bash
# Check if libraries are installed
ldconfig -p | grep ucx
ldconfig -p | grep nixl

# If not found, update ldconfig
sudo ldconfig

# Or add library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Problem**: UCX configure fails with "CUDA not found"

```bash
# Make sure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Then re-run configure
```

**Problem**: Meson not found

```bash
# Install meson in user directory
pip3 install --user meson ninja

# Make sure ~/.local/bin is in PATH
export PATH=$HOME/.local/bin:$PATH
```

### Runtime Issues

**Problem**: "No CUDA-capable device is detected"

```bash
# Check NVIDIA driver
nvidia-smi

# If driver is not loaded, reinstall NVIDIA driver
```

**Problem**: "Insufficient GPUs" error

The benchmark requires at least 2 GPUs. If you have only 1 GPU, you can modify the code to use CUDA streams on a single GPU for testing (not representative of real performance).

## Verification

After installation, verify everything works:

```bash
# Test baseline
make baseline
./main

# Test with NCCL (if installed)
make nccl
./main

# Test with NIXL (if installed)
make nixl
./main

# Test with both (if both installed)
make full
./main
```

## Performance Tips

1. **Use NVLink**: For best performance, ensure GPUs are connected via NVLink
   ```bash
   nvidia-smi nvlink --status
   ```

2. **Enable Persistence Mode**: Reduces kernel launch latency
   ```bash
   sudo nvidia-smi -pm 1
   ```

3. **Set GPU Clocks**: Lock clocks for consistent benchmarking
   ```bash
   sudo nvidia-smi -lgc 1410,1410  # Adjust for your GPU
   ```

4. **Disable CPU Frequency Scaling**: For more consistent results
   ```bash
   sudo cpupower frequency-set --governor performance
   ```

## Additional Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NCCL GitHub](https://github.com/NVIDIA/nccl)
- [NIXL GitHub](https://github.com/ai-dynamo/nixl)
- [UCX Documentation](https://openucx.readthedocs.io/)
- [GDRCopy GitHub](https://github.com/NVIDIA/gdrcopy)
- [UCCL KV Transfer Engine Blog Post](https://uccl-project.github.io/posts/kv-transfer-engine/)

## Quick Reference

```bash
# Full installation from scratch
sudo apt-get install -y build-essential cmake pkg-config libnccl2 libnccl-dev
pip3 install --user meson ninja pybind11 tomlkit

# Build UCX
git clone https://github.com/openucx/ucx.git && cd ucx && git checkout v1.20.0
./autogen.sh && ./configure --with-cuda=/usr/local/cuda --enable-mt
make -j$(nproc) && sudo make install && sudo ldconfig

# Build NIXL
git clone https://github.com/ai-dynamo/nixl.git && cd nixl
meson setup build && cd build && ninja && sudo ninja install && sudo ldconfig

# Build and run benchmark
cd /home/ubuntu/cuda_examples/TensorParallelFromScratch/03_nccl_vs_nixl
# Edit Makefile to enable NCCL and NIXL flags
make full
./main
```

