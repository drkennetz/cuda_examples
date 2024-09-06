# CUDA Examples

Welcome to CUDA Examples! This project intends to serve as a practical landing page for CUDA C++ development. If you see anything present or missing that might make it more useful, open an issue.

## Structure

The repository is split into a few main structures by example type:
- [SetupAndInitExamples](./SetupAndInitExamples/) - stuff that happens at the beginning of a CUDA program (this is not installation).
- [MemoryAndStructureExmaples](./MemoryAndStructureExamples/) - examples related to ways to allocate memory, launch kernels, or structure code to be beneificial for use with CUDA. These can and will leverage code / kernel samples, but their theme will be more around "good ways to do things" or "considerations" when writing CUDA programs.
- [KernelAndLibExamples](./KernelAndLibExamples/) - kernels, core libraries, thrust, etc. Just general examples of how to actually load and process data on the GPU.
- [ProfilingExamples](./ProfilingExamples/) - examples to profile or benchmark CUDA code.
- [PerformanceChecklist](./PerformanceChecklistExamples/) - examples which cover the standard CUDA performance checklist.

## Contributing
I love contributions because I get to learn from you and the project grows to help more people. With that in mind, contributions should meet the following characteristics:
- Novel - no existing examples like them
- Documented - there should be a clear explanation of what you are conveying with the add
- Correct - Self-explanatory

I imagine that CUDA kernel samples, thrust samples, and other core library examples will fill up the most quickly under `KernelAndLibExamples`, which means that one will eventually be the hardest to contribute to. When forming a contribution, PLEASE ensure that you are showing something novel. I do not want to reject any PRs because they are redundant - you spent time on that! But alas, I will if they don't show new concepts.

### Conventions
As sort of a sub-section of contributions, I'd like to cover conventions to use:
- Each example should have its own subdirectory under the appropriate section
- If your example produces output, it should run as a bash script and the output files should be added to `.gitignore`
- All executables should be called `main` (this is also for `.gitignore`)
- All examples should work with C++20.
- Each example should include `utils/utils.cuh`, and each CUDA call that returns a `cudaError_t` should be wrapped with a `cudaCheckError` like `cudaCheckError(::cudaDeviceSynchronize());`.
- Each CUDA API call should be prefixed with `::` to inform the reader that it is an external API call. See example above.

## Building Instructions

The CUDA_ARCHITECTURES defaults to 86. To run with the default:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

If you require a single different architecture (and run with 8 processors):
```bash
mkdir build && cd build
cmake -DCUDA_ARCHITECTURES="80" ..
make -j8
```
If you'd like to pass multiple architectures:
```bash
mkdir build && cd build
cmake -DCUDA_ARCHITECTURES="80;86;90" ..
make -j8
```

To build a particular example:

```bash
cd <ExampleDir>/<example>
make -j$(nproc)
```
The make files are less robust and may need modified. Feel free to submit a PR to fix them!
