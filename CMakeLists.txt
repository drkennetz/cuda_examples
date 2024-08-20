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
ConfigureCUDAExample(nvtx_example ProfilingExamples/nvtx/nvtx_example.cu)
ConfigureCUDAExample(vector_add KernelAndLibExamples/vector_add/vector_add.cu)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ProfilingExamples/nvtx/make_run_and_profile.sh
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/make_run_and_profile.sh
    COPYONLY
)
