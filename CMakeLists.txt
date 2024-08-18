cmake_minimum_required(VERSION 3.29 FATAL_ERROR)
project(cuda_examples LANGUAGES CXX CUDA)

function(ConfigureCUDAExample CMAKE_CUDA_EXAMPLE_NAME)
  add_executable(${CMAKE_CUDA_EXAMPLE_NAME} ${ARGN})
  target_compile_options(
    ${CMAKE_CUDA_EXAMPLE_NAME} 
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>) 
endfunction()

ConfigureCUDAExample(setup_check 001_setup_check/hello_world.cu)
ConfigureCUDAExample(simple_thrust 002_simple_thrust/simple_thrust.cu)
ConfigureCUDAExample(bandwidth_check 003_bandwidth_check/bandwidth_check.cu)
ConfigureCUDAExample(nvtx_example 004_nvtx/nvtx_example.cu)
