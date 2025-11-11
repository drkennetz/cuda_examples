# Makefile for NCCL vs NIXL KV Cache Transfer Benchmark

# Compiler
NVCC = nvcc

# Target executable
TARGET = main

# Source files
SRC = nixl_vs_nccl.cu

# CUDA architecture (adjust for your GPU)
CUDA_ARCH = -arch=sm_86

# Compiler flags
CXXFLAGS = -std=c++20
NVCCFLAGS = $(CUDA_ARCH) $(CXXFLAGS) -O3

# Libraries and includes
LIBS = -lcudart
INCLUDES = -I../../utils

# Optional: NCCL support
# NCCL is installed in system paths (/usr/include, /lib/x86_64-linux-gnu)
NCCL_INCLUDE = -I/usr/include
NCCL_LIB = -lnccl
NCCL_FLAGS = -DUSE_NCCL
# Note: System library paths are automatically searched, so no -L needed

# Optional: NIXL support
# NIXL is installed at /opt/nvidia/nvda_nixl/
NIXL_HOME = /opt/nvidia/nvda_nixl
NIXL_INCLUDE = -I$(NIXL_HOME)/include
NIXL_LIB = -L$(NIXL_HOME)/lib/x86_64-linux-gnu -lnixl -lnixl_common -lnixl_build
NIXL_FLAGS = -DUSE_NIXL
# Note: NIXL requires C++ standard library linking and runtime library path
# Use -Xlinker to pass linker flags through NVCC
NIXL_LDFLAGS = -Xlinker -rpath -Xlinker $(NIXL_HOME)/lib/x86_64-linux-gnu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Baseline only (no NCCL/NIXL)
baseline: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(TARGET) $< $(LIBS)

# Build with NCCL only
nccl: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(NCCL_FLAGS) $(INCLUDES) $(NCCL_INCLUDE) -o $(TARGET) $< $(LIBS) $(NCCL_LIB)

# Build with NIXL only
nixl: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(NIXL_FLAGS) $(INCLUDES) $(NIXL_INCLUDE) -o $(TARGET) $< $(LIBS) $(NIXL_LIB) $(NIXL_LDFLAGS)

# Build with both NCCL and NIXL
full: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(NCCL_FLAGS) $(NIXL_FLAGS) $(INCLUDES) $(NCCL_INCLUDE) $(NIXL_INCLUDE) -o $(TARGET) $< $(LIBS) $(NCCL_LIB) $(NIXL_LIB) $(NIXL_LDFLAGS)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Build baseline only (cudaMemcpy)"
	@echo "  make baseline - Build baseline only"
	@echo "  make nccl     - Build with NCCL support"
	@echo "  make nixl     - Build with NIXL support"
	@echo "  make full     - Build with both NCCL and NIXL"
	@echo "  make run      - Build and run the benchmark"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Note: Adjust NCCL_HOME and NIXL_HOME paths in Makefile before building with those options"

.PHONY: all baseline nccl nixl full run clean help

