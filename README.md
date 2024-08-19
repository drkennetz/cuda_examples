# CUDA Examples

## Building Instructions

To build all the examples at once:

```bash
mkdir build/
cd build/
cmake ..
make -j$(nproc)
```

To build a particular example:

```bash
cd 00X_example/
make -j$(nproc)
```
