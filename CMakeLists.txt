cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_examples LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Allow CUDA_ARCHITECTURES to be set from the command line, default to 86 if not set.
set(CUDA_ARCHITECTURES_DEFAULT "86" CACHE STRING "CUDA architectures to compile for")
set(CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES_DEFAULT} CACHE STRING "CUDA architectures")
message(STATUS "Using CUDA_ARCHITECTURE: ${CUDA_ARCHITECTURES}")

function(ConfigureCUDAExample CMAKE_CUDA_EXAMPLE_NAME)
  add_executable(${CMAKE_CUDA_EXAMPLE_NAME} ${ARGN})
  target_link_libraries(${CMAKE_CUDA_EXAMPLE_NAME} PRIVATE nvToolsExt)
  target_compile_options(
    ${CMAKE_CUDA_EXAMPLE_NAME} 
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  set_target_properties(${CMAKE_CUDA_EXAMPLE_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
endfunction()

ConfigureCUDAExample(setup_check SetupAndInitExamples/setup_check/hello_world.cu)
ConfigureCUDAExample(simple_thrust KernelAndLibExamples/simple_thrust/simple_thrust.cu)
ConfigureCUDAExample(bandwidth_check ProfilingExamples/bandwidth_check/bandwidth_check.cu)
ConfigureCUDAExample(cuda_streams PerformanceChecklistExamples/cuda_streams/cuda_streams.cu)
ConfigureCUDAExample(nvtx_example ProfilingExamples/nvtx/nvtx_example.cu)
ConfigureCUDAExample(vector_add KernelAndLibExamples/vector_add/vector_add.cu)
ConfigureCUDAExample(alloc_init_vs_alloc_uninit MemoryAndStructureExamples/alloc_init_vs_alloc_uninit/alloc_init_vs_alloc_uninit.cu)
ConfigureCUDAExample(pinned_vs_pageable MemoryAndStructureExamples/pinned_vs_pageable/pinned_vs_pageable.cu)
ConfigureCUDAExample(bank_conflicts ProfilingExamples/memory_bank_conflict/bank_conflict.cu)
ConfigureCUDAExample(global_vs_shared ProfilingExamples/global_vs_shared_mem/global_vs_shared.cu)
ConfigureCUDAExample(matmul TensorParallelFromScratch/01_simple_matmul_no_tp/matmul.cu)
ConfigureCUDAExample(tp_matmul_small TensorParallelFromScratch/02_matmul_tp/matmul_tp_small.cu)
ConfigureCUDAExample(tp_matmul_big TensorParallelFromScratch/02_matmul_tp/matmul_tp_big.cu)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ProfilingExamples/nvtx/make_run_and_profile.sh
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/make_run_and_profile.sh
    COPYONLY
)
