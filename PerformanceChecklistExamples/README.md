# Performance Checklist

Topics every CUDA developer should be thinking about when they write an application.

### Coalesce Global Memory Accesses
- Global memory accesses are slow.
- 32 threads run in a warp. Each thread accesses a contiguous memory address that can be combined into a single memory transaction.
- Reduces the number of memory transactions required to satisfy warp's memory requests, improving performance.
- Modern GPUs have hardware that can help
- Memory accesses should be aligned to cache line boundaries
- Non-contiguous / strided accesses within a warp can signifcantly reduce effective memory bandwidth.
- Coalesced memory access can lead to significant performance improvements, potentially up to 10x faster than non-coalesced
- Organize data structures and access patterns to promote coalescing.
    - This often means structure of arrays instead of array of structures
- When coalesced global memory access is not possible, using shared memory as an intermediate step can help.

