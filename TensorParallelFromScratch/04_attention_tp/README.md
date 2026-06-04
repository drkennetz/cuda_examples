# Part 04 — Tensor-Parallel Attention From Scratch

For the full writeup, visit: *(substack link TBD)*

[Part 03](../03_standard_attention/) computed one attention head on one GPU. This
part shards **multi-head** attention across 1, 2, 4, or 8 GPUs using **genuine
tensor parallelism** — the same Megatron-LM scheme used to serve real LLMs — with
a hand-rolled all-reduce over NVLink. No NCCL, no libraries.

## Data parallel vs. tensor parallel

It's worth being precise, because it's easy to fake. In [Part 02](../02_matmul_dp/)
we split a matmul's output columns across GPUs; the columns were independent, so
no GPU needed to talk to any other — which is exactly why that example is named
*data* parallel, not tensor parallel. You could shard attention the same lazy way —
give each GPU a slice of the **query rows** — and since each output row is
computed independently, again nobody communicates. That's **data / sequence
parallelism**. It works, but it isn't *tensor* parallelism, and it doesn't teach
the thing that actually matters at scale: **the collective**.

Tensor parallelism splits a *single logical operation* across GPUs such that
recombining the result **requires communication** — an all-reduce or all-gather.
That communication is the whole game: it's what bounds how far you can shard a
layer before the network, not the math, becomes the limit.

## How real LLMs shard attention: head parallelism

Production transformers run `H` heads in parallel (here `H = 32`, each of width
`d = 128`, for a model dimension `DM = H·d = 4096`). Megatron-LM shards attention
by **partitioning the heads across GPUs**:

```
                 GPU 0            GPU 1           ...
heads:        [0 .. H/g)      [H/g .. 2H/g)
              ┌─────────┐      ┌─────────┐
per head:     S=QKᵀ/√d         S=QKᵀ/√d          ← fully independent,
              softmax          softmax              no communication
              O=PV             O=PV
              └────┬────┘      └────┬────┘
            O_local[N, F]    O_local[N, F]        F = (H/g)·d
                   │                │
        partial = O_local · W_g     ·              ← row-parallel output proj.
              [N,F]·[F,DM]           ·                W_O is split by ROWS
                   │                │
                   └──── ALL-REDUCE (sum) ────┘    ← the collective that makes
                          Y = Σ partialₘ            this *tensor* parallel
                              [N, DM]
```

Two stages:

1. **Per-head attention (no comms).** Each GPU owns `H/ngpus` heads and runs the
   full Part-03 three-pass attention for each one, writing the results
   side-by-side into a local `[N, F]` buffer (`F = (H/ngpus)·d`).
2. **Row-parallel output projection + all-reduce (the comms).** Every transformer
   wraps multi-head attention in an output projection `W_O` of shape `[DM, DM]`
   that mixes the heads back together. Because GPU `g` only holds *its* heads'
   features, it owns the corresponding **rows** of `W_O` and can only compute a
   **partial** `[N, DM]` result. The true output is the **sum** of all the
   partials — so we **all-reduce** them. This is the genuine TP collective; it
   exists precisely because `W_O` couples features that live on different GPUs.

The answer is bit-for-bit **invariant to `ngpus`** (a partitioned contraction is
still the same contraction) — TP changes *where* the math runs, not the result.

### Why head parallelism, and not "split the single head"?

You could keep Part 03's single head and split its `d`-contraction across GPUs.
But softmax needs a whole row of `S`, so you'd have to **all-reduce the entire
`N×N = 4096×4096` score matrix** (32 MiB) *every layer* — a huge collective on an
intermediate that head parallelism never has to move. The only thing head
parallelism communicates is the `[N, DM]` projection output. That's why real
systems shard by heads, and it's a nice illustration of choosing the parallelism
axis to minimize the collective.

## The all-reduce (from scratch, over NVLink)

This box is a full NVLink mesh (`nvidia-smi topo -m` shows `NV18` between every
pair), so peer-to-peer GPU copies are fast and direct. The all-reduce is
deliberately the simplest correct thing:

- Enable peer access between all GPU pairs (`cudaDeviceEnablePeerAccess`).
- Each GPU seeds an FP32 accumulator with its own partial, then pulls every
  peer's (immutable) partial via `cudaMemcpyPeerAsync` and adds it in.

That's `O(ngpus²)` transfers — fine for a teaching example up to 8 GPUs. A real
**ring all-reduce** moves `2·(ngpus-1)/ngpus` of the data per GPU regardless of
count; NCCL does this (plus tree/NVLS variants). Swapping this loop for a ring is
a good follow-on exercise. Partials cross the wire in FP16; the accumulator is
FP32 to avoid drift.

## Build and run

`Makefile` targets `sm_90` (H100). Pass the GPU count as an argument; it must
divide `H = 32` (so 1, 2, 4, 8, 16, 32 are valid). Defaults to all visible GPUs.

```bash
make
./main 1
./main 2
./main 4
./main 8
```

Or via the repo CMake build (`cmake -DCUDA_ARCHITECTURES=90 ..`): `./attention_tp 8`.

## Results (8×H100, this box)

| GPUs | heads/GPU | slowest compute | all-reduce | end-to-end | verify |
|-----:|----------:|----------------:|-----------:|-----------:|:------:|
| 1    | 32        | 209.3 ms        | 0.00 ms    | 209.3 ms   | PASS   |
| 2    | 16        | 104.9 ms        | 0.30 ms    | 105.4 ms   | PASS   |
| 4    | 8         |  52.4 ms        | 1.27 ms    |  54.1 ms   | PASS   |
| 8    | 4         |  26.3 ms        | 4.99 ms    |  32.1 ms   | PASS   |

The compute scales almost perfectly — `209 → 105 → 52 → 26 ms`, near-linear in the
GPU count, because the heads are independent. The all-reduce, meanwhile, **grows**
with the GPU count (`O(ngpus²)` here). That's the fundamental tensor-parallel
tradeoff in one table: you buy compute scaling and pay for it in communication,
and past some point the collective dominates. It's why TP is typically kept
within a single NVLink node and combined with pipeline/data parallelism across
nodes — and why a better collective (ring all-reduce) is the obvious next step.

**Still the naive attention kernels from [Part 03](../03_standard_attention/)** —
we sharded the work, we didn't speed up the math. Fusing the three passes
(FlashAttention) is a separate axis of improvement, orthogonal to the
parallelism shown here.
