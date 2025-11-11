#!/bin/bash

# Test script for NCCL vs NIXL baseline benchmark
# This script compiles and runs the baseline (cudaMemcpy only) version

set -e  # Exit on error

echo "==========================================="
echo "NCCL vs NIXL Baseline Test"
echo "==========================================="

# Set CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Check CUDA availability
echo ""
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

nvcc --version

# Check for GPUs
echo ""
echo "Checking for GPUs..."
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo ""
    echo "Warning: This benchmark requires at least 2 GPUs for meaningful results."
    echo "You can still run it, but it will show an error message."
    echo ""
fi

# Build
echo ""
echo "Building baseline benchmark..."
make clean
make

# Check if build was successful
if [ ! -f "./main" ]; then
    echo "Error: Build failed. Executable not found."
    exit 1
fi

echo ""
echo "Build successful!"

# Run if requested
if [ "$1" == "--run" ] || [ "$1" == "-r" ]; then
    echo ""
    echo "Running benchmark..."
    echo "==========================================="
    ./main
else
    echo ""
    echo "To run the benchmark, use:"
    echo "  ./main"
    echo "or:"
    echo "  ./test_baseline.sh --run"
fi

echo ""
echo "==========================================="
echo "Test complete!"
echo "==========================================="

