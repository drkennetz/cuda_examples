// Standard ("naive") scaled dot-product attention, from scratch, on a single GPU.
//
// This is the textbook three-pass algorithm that materializes the full N x N
// score (S) and probability (P) matrices in HBM:
//
//   1. Load Q, K from HBM, compute S = (Q K^T) / sqrt(d), write S to HBM.
//   2. Read S from HBM, compute P = softmax(S) row-wise, write P to HBM.
//   3. Load P, V from HBM, compute O = P V, write O to HBM.
//   4. Return O.
//
// Shapes (single attention head):
//   Q, K, V : [N, d]   (N = sequence length, d = head dimension)
//   S, P    : [N, N]
//   O       : [N, d]
//
// Storage is FP16 (2 bytes / value, as in real LLM inference). All reductions
// (the QK^T dot products, the softmax sum, and the PV dot products) accumulate
// in FP32 to stay numerically sane over N = 4096 length sums. This is exactly
// how attention is done in practice, and it keeps our GPU result within a tight
// tolerance of the FP32 CPU reference.
//
// NOTE: this is intentionally the NAIVE version. The kernels use plain global
// memory access (no shared-memory tiling, no tensor cores). The whole point of
// the example is to show the HBM round-trips that FlashAttention later removes
// by fusing these three passes and never materializing S/P. See the README.

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../utils/utils.cuh"

#define N 4096  // sequence length
#define D 128   // head dimension

// ---------------------------------------------------------------------------
// Pass 1: S = (Q K^T) * scale
//   Q : [N, D], K : [N, D]  ->  S : [N, N]
//   One thread computes one S[row, col] = scale * sum_k Q[row,k] * K[col,k].
// ---------------------------------------------------------------------------
__global__ void qkKernel(const __half* __restrict__ Q,
                         const __half* __restrict__ K,
                         __half* __restrict__ S,
                         int n, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index i
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // key index   j

    if (row < n && col < n) {
        float acc = 0.0f;  // accumulate in FP32
        for (int k = 0; k < d; k++) {
            acc += __half2float(Q[row * d + k]) * __half2float(K[col * d + k]);
        }
        S[row * n + col] = __float2half(acc * scale);
    }
}

// ---------------------------------------------------------------------------
// Pass 2: P = softmax(S) along each row (the key dimension).
//   One thread block handles one row of length N. We do the numerically stable
//   softmax: subtract the row max before exp, then divide by the row sum.
//   Shared-memory parallel reductions are used for both the max and the sum.
// ---------------------------------------------------------------------------
__global__ void softmaxKernel(const __half* __restrict__ S,
                              __half* __restrict__ P,
                              int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ float sdata[];

    // --- row max ---
    float local_max = -INFINITY;
    for (int col = tid; col < n; col += nthreads) {
        local_max = fmaxf(local_max, __half2float(S[row * n + col]));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // --- row sum of exp(s - max) ---
    float local_sum = 0.0f;
    for (int col = tid; col < n; col += nthreads) {
        local_sum += __expf(__half2float(S[row * n + col]) - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float row_sum = sdata[0];
    float inv_sum = 1.0f / row_sum;
    __syncthreads();

    // --- normalize ---
    for (int col = tid; col < n; col += nthreads) {
        float p = __expf(__half2float(S[row * n + col]) - row_max) * inv_sum;
        P[row * n + col] = __float2half(p);
    }
}

// ---------------------------------------------------------------------------
// Pass 3: O = P V
//   P : [N, N], V : [N, D]  ->  O : [N, D]
//   One thread computes one O[row, k] = sum_j P[row,j] * V[j,k].
// ---------------------------------------------------------------------------
__global__ void pvKernel(const __half* __restrict__ P,
                         const __half* __restrict__ V,
                         __half* __restrict__ O,
                         int n, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index i
    int k   = blockIdx.x * blockDim.x + threadIdx.x;  // feature index

    if (row < n && k < d) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            acc += __half2float(P[row * n + j]) * __half2float(V[j * d + k]);
        }
        O[row * d + k] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// CPU reference (FP32) computed from the same FP16 inputs, for verification.
// ---------------------------------------------------------------------------
void attentionHost(const std::vector<float>& Q,
                   const std::vector<float>& K,
                   const std::vector<float>& V,
                   std::vector<float>& O,
                   int n, int d, float scale) {
    std::vector<float> S(n * n);
    for (int i = 0; i < n; i++) {
        // S row + stable softmax
        float row_max = -INFINITY;
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++) acc += Q[i * d + k] * K[j * d + k];
            S[i * n + j] = acc * scale;
            row_max = std::fmax(row_max, S[i * n + j]);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            float e = std::exp(S[i * n + j] - row_max);
            S[i * n + j] = e;
            row_sum += e;
        }
        for (int j = 0; j < n; j++) S[i * n + j] /= row_sum;
        // O row = P row . V
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += S[i * n + j] * V[j * d + k];
            O[i * d + k] = acc;
        }
    }
}

int main() {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::cout << "Standard (naive) attention, single GPU\n";
    std::cout << "  N = " << N << ", d = " << D << ", dtype = FP16 (FP32 accumulate)\n";
    std::cout << "  S, P materialized in HBM: "
              << (2.0 * N * N * sizeof(__half)) / (1024.0 * 1024.0) << " MiB total\n\n";

    // --- Host data (FP32 reference values + FP16 copies for the device) ---
    std::vector<float> hQ(N * D), hK(N * D), hV(N * D);
    std::srand(1234);
    auto rnd = [] { return (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f; };
    for (int i = 0; i < N * D; i++) { hQ[i] = rnd(); hK[i] = rnd(); hV[i] = rnd(); }

    std::vector<__half> hQ16(N * D), hK16(N * D), hV16(N * D);
    for (int i = 0; i < N * D; i++) {
        hQ16[i] = __float2half(hQ[i]);
        hK16[i] = __float2half(hK[i]);
        hV16[i] = __float2half(hV[i]);
    }
    // Use the rounded FP16 values for the CPU reference too, so we compare
    // against the same inputs the GPU actually sees.
    for (int i = 0; i < N * D; i++) {
        hQ[i] = __half2float(hQ16[i]);
        hK[i] = __half2float(hK16[i]);
        hV[i] = __half2float(hV16[i]);
    }

    // --- Device allocations ---
    __half *dQ, *dK, *dV, *dS, *dP, *dO;
    cudaCheckError(::cudaMalloc(&dQ, N * D * sizeof(__half)));
    cudaCheckError(::cudaMalloc(&dK, N * D * sizeof(__half)));
    cudaCheckError(::cudaMalloc(&dV, N * D * sizeof(__half)));
    cudaCheckError(::cudaMalloc(&dS, (size_t)N * N * sizeof(__half)));
    cudaCheckError(::cudaMalloc(&dP, (size_t)N * N * sizeof(__half)));
    cudaCheckError(::cudaMalloc(&dO, N * D * sizeof(__half)));

    cudaCheckError(::cudaMemcpy(dQ, hQ16.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    cudaCheckError(::cudaMemcpy(dK, hK16.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
    cudaCheckError(::cudaMemcpy(dV, hV16.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1, e2, e3;
    cudaCheckError(::cudaEventCreate(&e0));
    cudaCheckError(::cudaEventCreate(&e1));
    cudaCheckError(::cudaEventCreate(&e2));
    cudaCheckError(::cudaEventCreate(&e3));

    // --- Pass 1: S = QK^T * scale ---
    dim3 block1(16, 16);
    dim3 grid1((N + block1.x - 1) / block1.x, (N + block1.y - 1) / block1.y);
    cudaCheckError(::cudaEventRecord(e0));
    qkKernel<<<grid1, block1>>>(dQ, dK, dS, N, D, scale);
    cudaCheckError(::cudaGetLastError());

    // --- Pass 2: P = softmax(S) ---
    int sm_threads = 256;
    cudaCheckError(::cudaEventRecord(e1));
    softmaxKernel<<<N, sm_threads, sm_threads * sizeof(float)>>>(dS, dP, N);
    cudaCheckError(::cudaGetLastError());

    // --- Pass 3: O = PV ---
    dim3 block3(32, 8);
    dim3 grid3((D + block3.x - 1) / block3.x, (N + block3.y - 1) / block3.y);
    cudaCheckError(::cudaEventRecord(e2));
    pvKernel<<<grid3, block3>>>(dP, dV, dO, N, D);
    cudaCheckError(::cudaGetLastError());
    cudaCheckError(::cudaEventRecord(e3));
    cudaCheckError(::cudaEventSynchronize(e3));

    float t_qk = 0, t_sm = 0, t_pv = 0;
    cudaCheckError(::cudaEventElapsedTime(&t_qk, e0, e1));
    cudaCheckError(::cudaEventElapsedTime(&t_sm, e1, e2));
    cudaCheckError(::cudaEventElapsedTime(&t_pv, e2, e3));

    std::vector<__half> hO16(N * D);
    cudaCheckError(::cudaMemcpy(hO16.data(), dO, N * D * sizeof(__half), cudaMemcpyDeviceToHost));

    std::cout << "GPU phase timings:\n";
    std::cout << "  Pass 1  S = QK^T/sqrt(d) : " << t_qk << " ms\n";
    std::cout << "  Pass 2  P = softmax(S)   : " << t_sm << " ms\n";
    std::cout << "  Pass 3  O = PV           : " << t_pv << " ms\n";
    std::cout << "  total                    : " << (t_qk + t_sm + t_pv) << " ms\n\n";

    // --- CPU reference + verification ---
    std::cout << "Computing CPU reference (FP32)...\n";
    std::vector<float> hO_ref(N * D);
    auto c0 = std::chrono::high_resolution_clock::now();
    attentionHost(hQ, hK, hV, hO_ref, N, D, scale);
    auto c1 = std::chrono::high_resolution_clock::now();
    std::cout << "  CPU time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count()
              << " ms\n\n";

    double max_abs = 0.0, sum_abs = 0.0;
    for (int i = 0; i < N * D; i++) {
        double diff = std::fabs(__half2float(hO16[i]) - hO_ref[i]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
    }
    double mean_abs = sum_abs / (N * D);
    const double tol = 2e-3;  // generous for FP16 output storage
    std::cout << "Verification (GPU vs CPU):\n";
    std::cout << "  max abs diff  : " << max_abs << "\n";
    std::cout << "  mean abs diff : " << mean_abs << "\n";
    std::cout << "  result: " << (max_abs < tol ? "PASS" : "FAIL") << " (tol " << tol << ")\n";

    cudaCheckError(::cudaEventDestroy(e0));
    cudaCheckError(::cudaEventDestroy(e1));
    cudaCheckError(::cudaEventDestroy(e2));
    cudaCheckError(::cudaEventDestroy(e3));
    cudaCheckError(::cudaFree(dQ));
    cudaCheckError(::cudaFree(dK));
    cudaCheckError(::cudaFree(dV));
    cudaCheckError(::cudaFree(dS));
    cudaCheckError(::cudaFree(dP));
    cudaCheckError(::cudaFree(dO));

    return (max_abs < tol) ? 0 : 1;
}
