#!/bin/bash

# Script to run NCCL vs NIXL benchmark with proper environment setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "==========================================="
echo "NCCL vs NIXL Benchmark Runner"
echo "==========================================="
echo ""
echo "Environment Setup:"
echo "  CUDA: /usr/local/cuda"
echo "  NIXL: /opt/nvidia/nvda_nixl"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check if executable exists
if [ ! -f "./main" ]; then
    echo "Building benchmark..."
    make clean
    
    # Determine which version to build
    if [ "$1" == "baseline" ]; then
        echo "Building baseline only..."
        make baseline
    elif [ "$1" == "nccl" ]; then
        echo "Building with NCCL..."
        make nccl
    elif [ "$1" == "nixl" ]; then
        echo "Building with NIXL..."
        make nixl
    else
        echo "Building with both NCCL and NIXL..."
        make full
    fi
fi

echo ""
echo "Running benchmark..."
echo "==========================================="
./main

echo ""
echo "==========================================="
echo "Benchmark complete!"
echo "==========================================="

