# Part 03 — Standard (Naive) Attention From Scratch

For the full writeup, visit: *(substack link TBD)*

This is the third entry in the **Tensor Parallel From Scratch** series. Parts
[01](../01_simple_matmul_no_tp/) and [02](../02_matmul_dp/) built a matrix
multiply and then sharded it across GPUs. Attention is, at its heart, two matrix
multiplies with a softmax wedged between them — so it's the natural next step.

This part implements **standard scaled dot-product attention** on a *single* GPU,
the way it's written in the original paper: materialize the full score matrix in
memory, normalize it, then multiply by the values. It is deliberately the
**naive, from-scratch** version — no shared-memory tiling, no tensor cores, no
fused kernels, no libraries. That makes the memory traffic obvious, which is
exactly what the later FlashAttention-style optimizations exist to remove.

[Part 04](../04_attention_tp/) takes this baseline and shards it across multiple
GPUs with real tensor parallelism.

## What attention computes

Given a sequence of `N` tokens, attention lets every token mix in information
from every other token. Each token carries three vectors of width `d`:

- **Q** (query) — "what am I looking for?"
- **K** (key) — "what do I contain?"
- **V** (value) — "what do I pass on if attended to?"

Stacked over the sequence, `Q, K, V` are each `[N, d]`. The output `O` is `[N, d]`:

```
        ┌─ scores ─┐   ┌─ probabilities ─┐
S = Q · Kᵀ / √d      P = softmax(S)        O = P · V
[N,d]·[d,N] = [N,N]  [N,N] (row-wise)      [N,N]·[N,d] = [N,d]
```

1. **`S = Q Kᵀ / √d`** — the dot product of every query with every key gives an
   `N×N` grid of similarity scores. The `1/√d` scaling keeps the scores from
   growing with `d` (which would saturate the softmax into a one-hot).
2. **`P = softmax(S)`** — applied independently to each **row**, turning each
   query's scores into a probability distribution over the keys. We use the
   numerically stable form: subtract the row max before `exp`, then divide by
   the row sum.
3. **`O = P V`** — each output row is a weighted average of all value vectors,
   weighted by that query's attention probabilities.

The softmax is the interesting part. The two matmuls are embarrassingly parallel,
but the softmax **couples an entire row** together — you need the whole row of
scores before you can normalize any of it. That coupling is what makes attention
harder to tile and shard than a plain matmul, and it's the reason FlashAttention
had to be clever (more below).

## The algorithm in this example

We follow the literal three-pass HBM algorithm:

| Pass | Kernel          | Reads          | Writes      | Work        |
|------|-----------------|----------------|-------------|-------------|
| 1    | `qkKernel`      | Q, K           | S → HBM     | `N·N·d`     |
| 2    | `softmaxKernel` | S ← HBM        | P → HBM     | `N·N`       |
| 3    | `pvKernel`      | P ← HBM, V     | O → HBM     | `N·N·d`     |

Configuration (single head): `N = 4096`, `d = 128`, **FP16** storage with **FP32
accumulation**. Every value in HBM is 2 bytes, but the dot products and the
softmax sums accumulate in 32-bit registers — standard practice that keeps the
result within `~1e-6` of the FP32 CPU reference while halving memory.

Note the cost of being naive: `S` and `P` are each `4096×4096×2B = 32 MiB`, so we
write and then re-read **64 MiB of intermediates** that never needed to exist.
For real sequence lengths this is the whole problem — the score matrix grows as
`N²`. At `N = 8192` it's 128 MiB per matrix; at `N = 32k` it's 2 GiB. This is the
quadratic-memory wall.

## Build and run

This box is 8×H100, so the `Makefile` targets `sm_90`. Change `-arch` for your
GPU (`sm_80` A100, `sm_86` RTX 30xx/A10, `sm_89` RTX 40xx).

```bash
make
./main
```

Example output (single H100):

```
Standard (naive) attention, single GPU
  N = 4096, d = 128, dtype = FP16 (FP32 accumulate)
  S, P materialized in HBM: 64 MiB total

GPU phase timings:
  Pass 1  S = QK^T/sqrt(d) : 4.78 ms
  Pass 2  P = softmax(S)   : 0.07 ms
  Pass 3  O = PV           : 0.90 ms
  total                    : 5.75 ms

Verification (GPU vs CPU):
  max abs diff  : 1.5602e-06
  result: PASS (tol 0.002)
```

Or via the repo CMake build: `cmake -DCUDA_ARCHITECTURES=90 ..` then run
`./standard_attention`.

## Why attention mattered, and how the hardware kept up

Attention is the core of the Transformer, and it scaled where recurrent models
couldn't: every token attends to every other token in **one parallel step**
instead of a sequential chain, which maps beautifully onto GPUs. That
parallelism is most of why LLMs became practical to train and serve at all.

But the `O(N²)` score matrix is both a memory and a bandwidth problem, and the
last several GPU generations are in large part a story of chasing it:

- **Volta (V100, 2017)** introduced **tensor cores** — dedicated matmul units
  that made the two GEMMs in attention dramatically faster, which immediately
  shifted the bottleneck onto the softmax and the HBM round-trips.
- **Ampere (A100, 2020)** added BF16 / TF32 and far more HBM bandwidth, and is
  the architecture where **FlashAttention** (Dao et al., 2022) landed:
  instead of materializing `S` and `P`, it **tiles** the computation and keeps
  the running softmax statistics (max and sum) in on-chip SRAM, fusing all three
  passes into one kernel. Same math, but the `N²` matrices never touch HBM — it's
  memory-IO-aware rather than compute-bound.
- **Hopper (H100, 2022)** added FP8 tensor cores, the Tensor Memory Accelerator
  (TMA) for asynchronous bulk copies, and thread-block clusters — which
  FlashAttention-3 uses to overlap the GEMMs with the softmax and push utilization
  much higher.
- **Blackwell (B200, 2024)** pushes further on FP8/FP4 and bandwidth for the
  ever-longer context windows.

Everything above is an optimization *on top of* the algorithm in this folder.
Here we write the textbook version, on purpose, so the later parts have a clear,
correct baseline to measure against — and so the HBM traffic that FlashAttention
removes is sitting right there in the timings.

**This is the naive implementation, from scratch.** Next: shard it across GPUs.
→ [Part 04 — Tensor-Parallel Attention](../04_attention_tp/)
