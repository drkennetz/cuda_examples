// Multi-head attention with TENSOR PARALLELISM across GPUs, from scratch.
//
// This is the next step after 03_standard_attention. Example 03 computed a
// single head on one GPU. Real transformers run many heads, and the canonical
// way to shard attention across GPUs is HEAD PARALLELISM -- the core
// head-parallel / row-parallel output-projection pattern used in Megatron-style
// tensor-parallel attention (we implement that pattern, not a full production
// stack -- there's no sharded QKV projection here, the Q/K/V are given):
//
//   * The H heads are partitioned across the GPUs. GPU g owns H/ngpus heads and
//     runs the *full* 3-pass attention (S = QK^T/sqrt(d), softmax, O = PV) for
//     each of its heads, completely independently. No communication yet.
//
//   * The per-head outputs are concatenated along the feature dimension and fed
//     through the attention OUTPUT PROJECTION W_O [DM, DM], where DM = H*d.
//     W_O is ROW-PARALLEL: GPU g holds the rows of W_O corresponding to its
//     heads' features and produces a PARTIAL [N, DM] result. The partials are
//     summed with an ALL-REDUCE across GPUs to form the final output.
//
// That all-reduce is what makes this genuine *tensor* parallelism (one logical
// matmul split across devices, recombined with a collective) rather than data
// parallelism (independent rows, no communication). We hand-roll the all-reduce
// with peer-to-peer (cudaMemcpyPeer) copies -- no NCCL, no libraries -- fastest
// when the GPUs are P2P/NVLink connected.
//
// Run:   ./main [ngpus] [--no-verify]   (ngpus must divide H; defaults to all)
//   e.g. ./main 1  /  ./main 2  /  ./main 4  /  ./main 8 --no-verify
//
// Precision: Q, K, V, S, P and the per-head output O are stored FP16 (2 bytes,
// as in real LLM inference), with all dot products / softmax accumulated in
// FP32. The output-projection partials and the all-reduce run entirely in FP32,
// so the collective itself adds no extra rounding. The math is the same
// contraction for any ngpus; in finite precision (summation is not associative,
// and the reduce order differs per device) we verify against the FP32 CPU
// reference within a tolerance rather than expecting a bit-exact match.

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../utils/utils.cuh"

#define N 4096   // sequence length
#define D 128    // per-head dimension
#define H 32     // number of attention heads  -> model dim DM = H*D = 4096
#define DM (H * D)

// ---------------------------------------------------------------------------
// Per-head attention kernels (same math as 03_standard_attention).
// ---------------------------------------------------------------------------

// S = (Q K^T) * scale, for one head.  Q,K : [N, D] -> S : [N, N]
__global__ void qkKernel(const __half* __restrict__ Q,
                         const __half* __restrict__ K,
                         __half* __restrict__ S,
                         int n, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < d; k++)
            acc += __half2float(Q[row * d + k]) * __half2float(K[col * d + k]);
        S[row * n + col] = __float2half(acc * scale);
    }
}

// P = softmax(S) row-wise (one block per row, numerically stable).
// NOTE: the shared-memory reductions assume blockDim.x is a power of two.
__global__ void softmaxKernel(const __half* __restrict__ S,
                              __half* __restrict__ P, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    extern __shared__ float sdata[];

    float local_max = -INFINITY;
    for (int col = tid; col < n; col += nthreads)
        local_max = fmaxf(local_max, __half2float(S[row * n + col]));
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < n; col += nthreads)
        local_sum += expf(__half2float(S[row * n + col]) - row_max);
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int col = tid; col < n; col += nthreads) {
        float p = expf(__half2float(S[row * n + col]) - row_max) * inv_sum;
        P[row * n + col] = __float2half(p);
    }
}

// O = P V for one head, written into a column slice of the concatenated
// O_local buffer:  Oout[row * out_stride + col_off + k].
__global__ void pvKernel(const __half* __restrict__ P,
                         const __half* __restrict__ V,
                         __half* __restrict__ Oout,
                         int n, int d, int out_stride, int col_off) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && k < d) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++)
            acc += __half2float(P[row * n + j]) * __half2float(V[j * d + k]);
        Oout[row * out_stride + col_off + k] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Row-parallel output projection:  Ypart = O_local @ W_g
//   O_local : [N, F]   (F = heads_on_this_gpu * D, this GPU's feature slice; FP16)
//   W_g     : [F, DM]  (the rows of W_O owned by this GPU; FP16)
//   Ypart   : [N, DM]  (FP32 partial result, summed across GPUs by the all-reduce)
// The partial is FP32: the collective sums FP32 values, so the all-reduce adds
// no rounding of its own and the only cross-ngpus difference is summation order.
// ---------------------------------------------------------------------------
__global__ void projKernel(const __half* __restrict__ O_local,
                           const __half* __restrict__ W_g,
                           float* __restrict__ Ypart,
                           int n, int f, int dm) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < dm) {
        float acc = 0.0f;
        for (int t = 0; t < f; t++)
            acc += __half2float(O_local[row * f + t]) * __half2float(W_g[t * dm + col]);
        Ypart[row * dm + col] = acc;
    }
}

// All-reduce helper: add a peer's FP32 partial into the FP32 accumulator.
__global__ void addFloat(float* __restrict__ acc, const float* __restrict__ add, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += add[i];
}

// ---------------------------------------------------------------------------
// Per-GPU device buffers.
// ---------------------------------------------------------------------------
struct GpuCtx {
    int   device;
    int   head0;        // first head owned by this GPU
    int   heads;        // number of heads owned (= H / ngpus)
    int   F;            // feature slice width = heads * D
    __half *Q, *K, *V;  // [heads, N, D] head-major
    __half *S, *P;      // [N, N] scratch, reused across heads
    __half *O_local;    // [N, F]
    __half *W_g;        // [F, DM]
    float  *partial;    // [N, DM]  (this GPU's FP32 partial projection)
    float  *staging;    // [N, DM]  (peer partials land here during all-reduce)
    float  *result;     // [N, DM]  (FP32 all-reduce accumulator)
    cudaStream_t stream;
    cudaEvent_t  start, stop;
};

int main(int argc, char** argv) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    int avail = 0;
    cudaCheckError(::cudaGetDeviceCount(&avail));
    int ngpus = avail;
    bool do_verify = true;
    for (int a = 1; a < argc; a++) {
        if (std::strcmp(argv[a], "--no-verify") == 0) do_verify = false;
        else ngpus = std::atoi(argv[a]);
    }
    if (ngpus < 1 || ngpus > avail) {
        std::cerr << "Requested " << ngpus << " GPUs but only " << avail << " visible.\n";
        return 1;
    }
    if (H % ngpus != 0) {
        std::cerr << "ngpus (" << ngpus << ") must divide H (" << H << ").\n";
        return 1;
    }
    const int heads_per_gpu = H / ngpus;
    const int F = heads_per_gpu * D;

    std::cout << "Tensor-parallel multi-head attention (head-parallel + W_O all-reduce)\n";
    std::cout << "  N = " << N << ", d = " << D << ", H = " << H << " heads, DM = " << DM << "\n";
    std::cout << "  GPUs = " << ngpus << ", heads/GPU = " << heads_per_gpu
              << ", feature slice F = " << F << "\n\n";

    // --- Host inputs (FP32 reference + FP16 device copies) ---
    // Q,K,V laid out head-major: [H, N, D]. W_O is [DM, DM].
    std::srand(1234);
    auto rnd = [] { return (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f; };

    std::vector<float>  hQ((size_t)H * N * D), hK((size_t)H * N * D), hV((size_t)H * N * D);
    std::vector<float>  hWo((size_t)DM * DM);
    for (auto& x : hQ) x = rnd();
    for (auto& x : hK) x = rnd();
    for (auto& x : hV) x = rnd();
    for (auto& x : hWo) x = rnd();

    // FP16 device-input copies in PINNED (page-locked) host memory so the per-GPU
    // H2D transfers below are genuinely async (overlap across streams) and run at
    // full bandwidth. The FP32 buffers above are CPU-reference-only and never
    // cross the bus, so they stay pageable.
    __half *hQ16, *hK16, *hV16, *hWo16;
    cudaCheckError(::cudaMallocHost(&hQ16,  hQ.size()  * sizeof(__half)));
    cudaCheckError(::cudaMallocHost(&hK16,  hK.size()  * sizeof(__half)));
    cudaCheckError(::cudaMallocHost(&hV16,  hV.size()  * sizeof(__half)));
    cudaCheckError(::cudaMallocHost(&hWo16, hWo.size() * sizeof(__half)));
    for (size_t i = 0; i < hQ.size(); i++) { hQ16[i]=__float2half(hQ[i]); hK16[i]=__float2half(hK[i]); hV16[i]=__float2half(hV[i]); }
    for (size_t i = 0; i < hWo.size(); i++) hWo16[i] = __float2half(hWo[i]);
    // Re-read FP16-rounded values so the CPU reference uses identical inputs.
    for (size_t i = 0; i < hQ.size(); i++) { hQ[i]=__half2float(hQ16[i]); hK[i]=__half2float(hK16[i]); hV[i]=__half2float(hV16[i]); }
    for (size_t i = 0; i < hWo.size(); i++) hWo[i] = __half2float(hWo16[i]);

    // --- Enable peer access between every pair of participating GPUs ---
    // This example is meant to run over P2P/NVLink. cudaMemcpyPeerAsync still
    // works without it (staging through the host), just much slower -- so warn
    // rather than abort if a pair can't talk directly.
    int no_p2p_pairs = 0;
    for (int i = 0; i < ngpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        for (int j = 0; j < ngpus; j++) {
            if (i == j) continue;
            int can = 0;
            cudaCheckError(::cudaDeviceCanAccessPeer(&can, i, j));
            if (can) {
                cudaError_t e = ::cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    cudaCheckError(e);
            } else {
                no_p2p_pairs++;
            }
        }
    }
    if (no_p2p_pairs > 0)
        std::cerr << "WARNING: " << no_p2p_pairs << " GPU pair(s) lack direct P2P access; "
                  << "the all-reduce will stage through the host and the NVLink timings "
                  << "below will not be representative.\n\n";

    // --- Per-GPU setup ---
    std::vector<GpuCtx> ctx(ngpus);
    for (int g = 0; g < ngpus; g++) {
        GpuCtx& c = ctx[g];
        c.device = g;
        c.head0  = g * heads_per_gpu;
        c.heads  = heads_per_gpu;
        c.F      = F;
        cudaCheckError(::cudaSetDevice(g));
        cudaCheckError(::cudaStreamCreate(&c.stream));
        cudaCheckError(::cudaEventCreate(&c.start));
        cudaCheckError(::cudaEventCreate(&c.stop));

        cudaCheckError(::cudaMalloc(&c.Q, (size_t)c.heads * N * D * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.K, (size_t)c.heads * N * D * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.V, (size_t)c.heads * N * D * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.S, (size_t)N * N * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.P, (size_t)N * N * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.O_local, (size_t)N * F * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.W_g, (size_t)F * DM * sizeof(__half)));
        cudaCheckError(::cudaMalloc(&c.partial, (size_t)N * DM * sizeof(float)));
        cudaCheckError(::cudaMalloc(&c.staging, (size_t)N * DM * sizeof(float)));
        cudaCheckError(::cudaMalloc(&c.result,  (size_t)N * DM * sizeof(float)));

        // Q/K/V for this GPU's heads (contiguous head-major slice). The host
        // source is pinned (see above), so these cudaMemcpyAsync calls are truly
        // asynchronous and can overlap across the per-GPU streams. They are still
        // synchronized before the timer starts, so they aren't part of the
        // measured kernel path.
        size_t head_elems = (size_t)N * D;
        cudaCheckError(::cudaMemcpyAsync(c.Q, &hQ16[(size_t)c.head0 * head_elems],
                                         (size_t)c.heads * head_elems * sizeof(__half),
                                         cudaMemcpyHostToDevice, c.stream));
        cudaCheckError(::cudaMemcpyAsync(c.K, &hK16[(size_t)c.head0 * head_elems],
                                         (size_t)c.heads * head_elems * sizeof(__half),
                                         cudaMemcpyHostToDevice, c.stream));
        cudaCheckError(::cudaMemcpyAsync(c.V, &hV16[(size_t)c.head0 * head_elems],
                                         (size_t)c.heads * head_elems * sizeof(__half),
                                         cudaMemcpyHostToDevice, c.stream));

        // Row slice of W_O owned by this GPU: rows [head0*D, (head0+heads)*D).
        cudaCheckError(::cudaMemcpyAsync(c.W_g, &hWo16[(size_t)c.head0 * D * DM],
                                         (size_t)F * DM * sizeof(__half),
                                         cudaMemcpyHostToDevice, c.stream));
    }

    // Finish all setup copies before we start timing the compute + all-reduce.
    for (int g = 0; g < ngpus; g++) {
        cudaCheckError(::cudaSetDevice(g));
        cudaCheckError(::cudaStreamSynchronize(ctx[g].stream));
    }

    auto wall0 = std::chrono::steady_clock::now();

    // --- Phase 1: each GPU computes attention for its heads, then its partial
    //     projection.  Fully independent; launched on each GPU's own stream. ---
    dim3 bQK(16, 16), gQK((N + 15) / 16, (N + 15) / 16);
    dim3 bPV(32, 8),  gPV((D + 31) / 32, (N + 7) / 8);
    dim3 bPR(16, 16), gPR((DM + 15) / 16, (N + 15) / 16);
    int smThreads = 256;  // power of two -- required by softmaxKernel's reduction

    for (int g = 0; g < ngpus; g++) {
        GpuCtx& c = ctx[g];
        cudaCheckError(::cudaSetDevice(g));
        cudaCheckError(::cudaEventRecord(c.start, c.stream));
        for (int lh = 0; lh < c.heads; lh++) {
            const __half* Qh = c.Q + (size_t)lh * N * D;
            const __half* Kh = c.K + (size_t)lh * N * D;
            const __half* Vh = c.V + (size_t)lh * N * D;
            qkKernel<<<gQK, bQK, 0, c.stream>>>(Qh, Kh, c.S, N, D, scale);
            cudaCheckError(::cudaGetLastError());
            softmaxKernel<<<N, smThreads, smThreads * sizeof(float), c.stream>>>(c.S, c.P, N);
            cudaCheckError(::cudaGetLastError());
            pvKernel<<<gPV, bPV, 0, c.stream>>>(c.P, Vh, c.O_local, N, D, c.F, lh * D);
            cudaCheckError(::cudaGetLastError());
        }
        // Partial output projection: O_local [N,F] @ W_g [F,DM] -> partial [N,DM]
        projKernel<<<gPR, bPR, 0, c.stream>>>(c.O_local, c.W_g, c.partial, N, c.F, DM);
        cudaCheckError(::cudaGetLastError());
        // Seed the all-reduce accumulator with this GPU's own (FP32) partial.
        cudaCheckError(::cudaMemcpyAsync(c.result, c.partial, (size_t)N * DM * sizeof(float),
                                         cudaMemcpyDeviceToDevice, c.stream));
        cudaCheckError(::cudaEventRecord(c.stop, c.stream));
    }
    // Make sure every GPU's partial is finished before we start exchanging them.
    for (int g = 0; g < ngpus; g++) {
        cudaCheckError(::cudaSetDevice(g));
        cudaCheckError(::cudaStreamSynchronize(ctx[g].stream));
    }

    // --- Phase 2: hand-rolled all-reduce of the [N, DM] FP32 partials.
    //     Every GPU pulls each peer's (immutable) partial into its staging
    //     buffer and adds it into its accumulator. O(ngpus^2) transfers --
    //     simple and correct; a ring all-reduce would cut the traffic. ---
    size_t ndm = (size_t)N * DM;
    int tpb = 256; int nb = (ndm + tpb - 1) / tpb;
    auto ar0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpus; g++) {
        GpuCtx& c = ctx[g];
        cudaCheckError(::cudaSetDevice(g));
        for (int src = 0; src < ngpus; src++) {
            if (src == g) continue;
            cudaCheckError(::cudaMemcpyPeerAsync(c.staging, g, ctx[src].partial, src,
                                                 ndm * sizeof(float), c.stream));
            addFloat<<<nb, tpb, 0, c.stream>>>(c.result, c.staging, ndm);
            cudaCheckError(::cudaGetLastError());
        }
    }
    for (int g = 0; g < ngpus; g++) {
        cudaCheckError(::cudaSetDevice(g));
        cudaCheckError(::cudaStreamSynchronize(ctx[g].stream));
    }
    auto ar1 = std::chrono::steady_clock::now();
    auto wall1 = ar1;

    // Per-GPU attention+projection time (the parallel part) and all-reduce time.
    float maxCompute = 0.0f;
    for (int g = 0; g < ngpus; g++) {
        float t = 0;
        cudaCheckError(::cudaEventElapsedTime(&t, ctx[g].start, ctx[g].stop));
        std::cout << "  GPU " << g << " attention+projection: " << t << " ms\n";
        maxCompute = std::max(maxCompute, t);
    }
    std::cout << "  slowest GPU compute      : " << maxCompute << " ms\n";
    std::cout << "  all-reduce (P2P/NVLink)  : "
              << std::chrono::duration<double, std::milli>(ar1 - ar0).count() << " ms\n";
    std::cout << "  end-to-end (compute+AR)  : "
              << std::chrono::duration<double, std::milli>(wall1 - wall0).count() << " ms\n\n";

    if (!do_verify) {
        std::cout << "Verification skipped (--no-verify).\n";
        // (cleanup below)
    }

    int rc = 0;
    if (do_verify) {
        // --- Pull the final result from GPU 0 (all GPUs hold the full sum) ---
        // Pinned host buffer for a full-bandwidth D2H copy of the [N, DM] result.
        const size_t n_out = (size_t)N * DM;
        float* hY;
        cudaCheckError(::cudaMallocHost(&hY, n_out * sizeof(float)));
        cudaCheckError(::cudaSetDevice(0));
        cudaCheckError(::cudaMemcpy(hY, ctx[0].result, n_out * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        // --- CPU reference (FP32): full multi-head attention + W_O projection ---
        std::cout << "Computing CPU reference (FP32)...\n";
        auto c0 = std::chrono::steady_clock::now();
        std::vector<float> Oconcat((size_t)N * DM, 0.0f);  // [N, DM] concatenated heads
        std::vector<float> Srow(N);
        for (int h = 0; h < H; h++) {
            const float* Qh = &hQ[(size_t)h * N * D];
            const float* Kh = &hK[(size_t)h * N * D];
            const float* Vh = &hV[(size_t)h * N * D];
            for (int i = 0; i < N; i++) {
                float row_max = -INFINITY;
                for (int j = 0; j < N; j++) {
                    float acc = 0.0f;
                    for (int k = 0; k < D; k++) acc += Qh[i * D + k] * Kh[j * D + k];
                    Srow[j] = acc * scale;
                    row_max = std::fmax(row_max, Srow[j]);
                }
                float rsum = 0.0f;
                for (int j = 0; j < N; j++) { Srow[j] = std::exp(Srow[j] - row_max); rsum += Srow[j]; }
                float inv = 1.0f / rsum;
                for (int k = 0; k < D; k++) {
                    float acc = 0.0f;
                    for (int j = 0; j < N; j++) acc += Srow[j] * inv * Vh[j * D + k];
                    Oconcat[(size_t)i * DM + h * D + k] = acc;
                }
            }
        }
        // Y = Oconcat [N,DM] @ W_O [DM,DM]
        std::vector<float> Yref((size_t)N * DM, 0.0f);
        for (int i = 0; i < N; i++)
            for (int t = 0; t < DM; t++) {
                float o = Oconcat[(size_t)i * DM + t];
                if (o == 0.0f) continue;
                for (int col = 0; col < DM; col++)
                    Yref[(size_t)i * DM + col] += o * hWo[(size_t)t * DM + col];
            }
        auto c1 = std::chrono::steady_clock::now();
        std::cout << "  CPU time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count() << " ms\n\n";

        double max_abs = 0.0, sum_abs = 0.0;
        for (size_t i = 0; i < n_out; i++) {
            double diff = std::fabs((double)hY[i] - (double)Yref[i]);
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        const double tol = 5e-2;  // FP16 storage of O_local + W_O, accumulated over DM
        std::cout << "Verification (GPU all-reduced result vs CPU):\n";
        std::cout << "  max abs diff  : " << max_abs << "\n";
        std::cout << "  mean abs diff : " << sum_abs / n_out << "\n";
        std::cout << "  result: " << (max_abs < tol ? "PASS" : "FAIL")
                  << " (tol " << tol << ", same mathematical contraction across ngpus)\n";
        rc = (max_abs < tol) ? 0 : 1;
        cudaCheckError(::cudaFreeHost(hY));
    }

    // --- Cleanup ---
    for (int g = 0; g < ngpus; g++) {
        GpuCtx& c = ctx[g];
        cudaCheckError(::cudaSetDevice(g));
        cudaCheckError(::cudaFree(c.Q)); cudaCheckError(::cudaFree(c.K)); cudaCheckError(::cudaFree(c.V));
        cudaCheckError(::cudaFree(c.S)); cudaCheckError(::cudaFree(c.P));
        cudaCheckError(::cudaFree(c.O_local)); cudaCheckError(::cudaFree(c.W_g));
        cudaCheckError(::cudaFree(c.partial)); cudaCheckError(::cudaFree(c.staging));
        cudaCheckError(::cudaFree(c.result));
        cudaCheckError(::cudaEventDestroy(c.start)); cudaCheckError(::cudaEventDestroy(c.stop));
        cudaCheckError(::cudaStreamDestroy(c.stream));
    }
    cudaCheckError(::cudaFreeHost(hQ16));
    cudaCheckError(::cudaFreeHost(hK16));
    cudaCheckError(::cudaFreeHost(hV16));
    cudaCheckError(::cudaFreeHost(hWo16));
    return rc;
}
