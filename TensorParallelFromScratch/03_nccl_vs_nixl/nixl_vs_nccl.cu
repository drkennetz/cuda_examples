/**
 * @file nixl_vs_nccl.cu
 * @brief Comparison of NCCL vs NIXL for KV Cache Transfer Engine
 * 
 * This example replicates the KV transfer engine concepts from:
 * https://uccl-project.github.io/posts/kv-transfer-engine/
 * 
 * Key differences:
 * - NCCL: Uses GPU SM resources for P2P transfers, higher latency
 * - NIXL: Offloads transfers from GPU, lower latency, no SM consumption
 * 
 * The benchmark simulates LLM inference KV cache transfers between GPUs.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#ifdef USE_NIXL
#include <nixl.h>
#endif

#include "../../utils/utils.cuh"

// KV cache configuration (typical for LLM inference)
struct KVCacheConfig {
    size_t num_layers = 32;          // Number of transformer layers
    size_t num_heads = 32;           // Number of attention heads
    size_t head_dim = 128;           // Dimension per head
    size_t seq_len = 2048;           // Sequence length
    size_t batch_size = 1;           // Batch size for inference
    
    size_t get_cache_size_per_layer() const {
        // K and V cache: [batch, num_heads, seq_len, head_dim]
        return 2 * batch_size * num_heads * seq_len * head_dim * sizeof(float);
    }
    
    size_t get_total_cache_size() const {
        return num_layers * get_cache_size_per_layer();
    }
};

// Helper macro for NCCL error checking
#ifdef USE_NCCL
#define ncclCheckError(call)                                                           \
    {                                                                                  \
        ncclResult_t status = call;                                                    \
        if (status != ncclSuccess) {                                                   \
            fprintf(stderr, "ERROR: NCCL call \"%s\" in line %d of file %s failed "   \
                    "with %s (%d).\n", #call, __LINE__, __FILE__,                      \
                    ncclGetErrorString(status), status);                               \
            exit(1);                                                                   \
        }                                                                              \
    }
#endif

// Helper macro for NIXL error checking (will be redefined in USE_NIXL section)

// Simple kernel to simulate computation on GPU
__global__ void simulateComputation(float* data, size_t size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
        }
        data[idx] = val;
    }
}

// Initialize KV cache with dummy data
void initializeKVCache(float* h_cache, size_t size) {
    for (size_t i = 0; i < size; i++) {
        h_cache[i] = static_cast<float>(i % 100) / 100.0f;
    }
}

// Benchmark result structure
struct BenchmarkResult {
    double avg_latency_ms;
    double throughput_gbps;
    double min_latency_ms;
    double max_latency_ms;
};

#ifdef USE_NCCL
/**
 * @brief Benchmark KV cache transfer using NCCL
 * 
 * NCCL uses GPU kernels for P2P communication, which consumes SM resources.
 * This can interfere with inference computation.
 */
BenchmarkResult benchmarkNCCL(const KVCacheConfig& config, int num_gpus, int iterations) {
    std::cout << "\n=== NCCL Benchmark ===" << std::endl;
    std::cout << "Note: NCCL P2P uses GPU SM resources" << std::endl;
    
    // Initialize NCCL communicators
    // For single-process multi-GPU, use ncclCommInitAll instead of ncclCommInitRank
    // ncclCommInitRank blocks until all ranks call it, which can cause hangs
    ncclComm_t comms[num_gpus];
    int devs[num_gpus];
    
    // Set up device array for ncclCommInitAll
    for (int i = 0; i < num_gpus; i++) {
        devs[i] = i;
    }
    
    std::cout << "Initializing NCCL communicators for " << num_gpus << " GPUs..." << std::endl;
    // ncclCommInitAll initializes all communicators at once (single-process multi-GPU)
    ncclCheckError(ncclCommInitAll(comms, num_gpus, devs));
    std::cout << "NCCL communicators initialized successfully" << std::endl;
    
    // Allocate KV cache on each GPU
    size_t cache_size = config.get_total_cache_size();
    size_t num_floats = cache_size / sizeof(float);
    
    float** d_cache = new float*[num_gpus];
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    cudaEvent_t* start_events = new cudaEvent_t[num_gpus];
    cudaEvent_t* end_events = new cudaEvent_t[num_gpus];
    
    std::cout << "Allocating cache size: " << cache_size << " bytes per GPU" << std::endl;
    std::cout << "Allocating " << num_gpus << " GPUs" << std::endl;

    for (int i = 0; i < num_gpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        cudaCheckError(::cudaMalloc(&d_cache[i], cache_size));
        cudaCheckError(::cudaStreamCreate(&streams[i]));
        cudaCheckError(::cudaEventCreate(&start_events[i]));
        cudaCheckError(::cudaEventCreate(&end_events[i]));
        
        // Initialize with dummy data
        cudaCheckError(::cudaMemset(d_cache[i], 0, cache_size));
    }
    
    // Warm-up
    std::cout << "Warming up for 5 iterations..." << std::endl;
    for (int iter = 0; iter < 5; iter++) {
        // Ring transfer pattern: GPU i sends to GPU (i+1), receives from GPU (i-1)
        // This simulates KV cache forwarding in distributed inference
        ncclGroupStart();
        for (int i = 0; i < num_gpus; i++) {
            int src = i;
            int dst = (i + 1) % num_gpus;
            int recv_from = (i - 1 + num_gpus) % num_gpus;
            cudaCheckError(::cudaSetDevice(src));
            ncclCheckError(ncclSend(d_cache[src], num_floats, ncclFloat, dst, 
                                   comms[src], streams[src]));
            ncclCheckError(ncclRecv(d_cache[src], num_floats, ncclFloat, 
                                   recv_from, comms[src], streams[src]));
        }
        ncclGroupEnd();
        
        for (int i = 0; i < num_gpus; i++) {
            cudaCheckError(::cudaSetDevice(i));
            cudaCheckError(::cudaStreamSynchronize(streams[i]));
        }
    }
    
    std::cout << "Benchmarking for " << iterations << " iterations..." << std::endl;
    // Benchmark
    std::vector<double> latencies;
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        // Record start time
        cudaCheckError(::cudaSetDevice(0));
        cudaCheckError(::cudaEventRecord(start_events[0], streams[0]));
        
        // Perform ring-based KV cache transfer
        // Ring pattern: GPU i sends to GPU (i+1), receives from GPU (i-1)
        ncclGroupStart();
        for (int i = 0; i < num_gpus; i++) {
            int src = i;
            int dst = (i + 1) % num_gpus;
            int recv_from = (i - 1 + num_gpus) % num_gpus;
            cudaCheckError(::cudaSetDevice(src));
            ncclCheckError(ncclSend(d_cache[src], num_floats, ncclFloat, dst, 
                                   comms[src], streams[src]));
            ncclCheckError(ncclRecv(d_cache[src], num_floats, ncclFloat, 
                                   recv_from, comms[src], streams[src]));
        }
        ncclGroupEnd();
        
        // Record end time
        cudaCheckError(::cudaSetDevice(0));
        cudaCheckError(::cudaEventRecord(end_events[0], streams[0]));
        
        // Wait for completion
        for (int i = 0; i < num_gpus; i++) {
            cudaCheckError(::cudaSetDevice(i));
            cudaCheckError(::cudaStreamSynchronize(streams[i]));
        }
        
        // Calculate latency
        float elapsed_ms;
        cudaCheckError(::cudaEventElapsedTime(&elapsed_ms, start_events[0], end_events[0]));
        latencies.push_back(elapsed_ms);
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();
    
    // Calculate statistics
    double sum = 0.0, min_lat = latencies[0], max_lat = latencies[0];
    for (double lat : latencies) {
        sum += lat;
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
    }
    double avg_latency = sum / latencies.size();
    
    // Calculate throughput (GB/s)
    double data_transferred_gb = (cache_size * iterations * num_gpus) / (1024.0 * 1024.0 * 1024.0);
    double throughput_gbps = data_transferred_gb / (total_time_ms / 1000.0);
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        ncclCommDestroy(comms[i]);
        cudaCheckError(::cudaFree(d_cache[i]));
        cudaCheckError(::cudaStreamDestroy(streams[i]));
        cudaCheckError(::cudaEventDestroy(start_events[i]));
        cudaCheckError(::cudaEventDestroy(end_events[i]));
    }
    
    delete[] d_cache;
    delete[] streams;
    delete[] start_events;
    delete[] end_events;
    
    return {avg_latency, throughput_gbps, min_lat, max_lat};
}
#endif

#ifdef USE_NIXL
// Helper macro for NIXL error checking
// Note: NIXL_IN_PROG (1) is not an error - it means operation is in progress
// Only negative values indicate errors
#define nixlCheckError(call, msg)                                                           \
    {                                                                                        \
        nixl_status_t status = call;                                                        \
        if (status < NIXL_SUCCESS) {                                                         \
            fprintf(stderr, "ERROR: NIXL call \"%s\" in line %d of file %s failed "        \
                    "with status %d. %s\n", #call, __LINE__, __FILE__, status, msg);       \
            exit(1);                                                                         \
        }                                                                                    \
    }

/**
 * @brief Benchmark KV cache transfer using NIXL
 * 
 * NIXL offloads transfers from GPU, freeing SM resources for computation.
 * This is ideal for overlapping communication with computation in inference.
 * 
 * Based on: https://github.com/ai-dynamo/nixl/blob/main/examples/cpp/nixl_example.cpp
 */
BenchmarkResult benchmarkNIXL(const KVCacheConfig& config, int num_gpus, int iterations) {
    std::cout << "\n=== NIXL Benchmark ===" << std::endl;
    std::cout << "Note: NIXL P2P does NOT use GPU SM resources" << std::endl;
    
    size_t cache_size = config.get_total_cache_size();
    
    // Allocate GPU memory for KV cache on each GPU
    float** d_cache = new float*[num_gpus];
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    cudaEvent_t* start_events = new cudaEvent_t[num_gpus];
    cudaEvent_t* end_events = new cudaEvent_t[num_gpus];
    
    for (int i = 0; i < num_gpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        cudaCheckError(::cudaMalloc(&d_cache[i], cache_size));
        cudaCheckError(::cudaStreamCreate(&streams[i]));
        cudaCheckError(::cudaEventCreate(&start_events[i]));
        cudaCheckError(::cudaEventCreate(&end_events[i]));
        cudaCheckError(::cudaMemset(d_cache[i], 0, cache_size));
    }
    
    std::cout << "Allocating cache size: " << cache_size << " bytes per GPU" << std::endl;
    std::cout << "Initializing NIXL agents for " << num_gpus << " GPUs..." << std::endl;
    
    // Create NIXL agents for each GPU
    std::vector<nixlAgent*> agents(num_gpus);
    std::vector<nixlBackendH*> backends(num_gpus);
    std::vector<nixl_opt_args_t> extra_params(num_gpus);
    
    nixlAgentConfig cfg(true);  // Enable GPU support
    
    for (int i = 0; i < num_gpus; i++) {
        std::string agent_name = "Agent" + std::to_string(i);
        agents[i] = new nixlAgent(agent_name, cfg);
        
        // Get available plugins and use UCX backend
        std::vector<nixl_backend_t> plugins;
        nixlCheckError(agents[i]->getAvailPlugins(plugins), "Failed to get available plugins");
        
        std::string backend_name = "UCX";
        nixl_b_params_t init_params;
        nixl_mem_list_t mems;
        
        nixlCheckError(agents[i]->getPluginParams(backend_name, mems, init_params),
                      "Failed to get plugin params");
        
        // Create backend
        nixlCheckError(agents[i]->createBackend(backend_name, init_params, backends[i]),
                      "Failed to create backend");
        
        extra_params[i].backends.push_back(backends[i]);
    }
    
    std::cout << "Registering GPU memory with NIXL..." << std::endl;
    
    // Register GPU memory with NIXL for each agent
    // Note: nixl_reg_dlist_t requires constructor parameter, so we create them individually
    std::vector<nixl_reg_dlist_t*> reg_lists(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        // Ensure CUDA device is set before memory operations
        cudaCheckError(::cudaSetDevice(i));
        cudaCheckError(::cudaDeviceSynchronize());  // Ensure device is ready
        
        reg_lists[i] = new nixl_reg_dlist_t(VRAM_SEG);
        
        nixlBlobDesc blob_desc;
        // Use CUDA device pointer directly - NIXL should detect it as GPU memory
        blob_desc.addr = reinterpret_cast<uintptr_t>(d_cache[i]);
        blob_desc.len = cache_size;
        blob_desc.devId = i;  // Set device ID for GPU memory
        
        reg_lists[i]->addDesc(blob_desc);
        
        // Register memory with the backend
        nixlCheckError(agents[i]->registerMem(*reg_lists[i], &extra_params[i]),
                      "Failed to register memory");
        
        // Verify device context is still set
        int current_device;
        cudaCheckError(::cudaGetDevice(&current_device));
        if (current_device != i) {
            fprintf(stderr, "Warning: Device context changed from %d to %d\n", i, current_device);
            cudaCheckError(::cudaSetDevice(i));
        }
    }
    
    std::cout << "Exchanging metadata between agents..." << std::endl;
    
    // Exchange metadata between agents (required for transfers)
    std::vector<std::string> local_mds(num_gpus);
    std::vector<std::vector<std::string>> remote_mds(num_gpus);
    
    for (int i = 0; i < num_gpus; i++) {
        nixlCheckError(agents[i]->getLocalMD(local_mds[i]), "Failed to get local metadata");
    }
    
    // Each agent loads metadata from all other agents
    for (int i = 0; i < num_gpus; i++) {
        remote_mds[i].resize(num_gpus);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                std::string agent_name = "Agent" + std::to_string(j);
                std::string loaded_md;
                nixlCheckError(agents[i]->loadRemoteMD(local_mds[j], loaded_md),
                              "Failed to load remote metadata");
                remote_mds[i][j] = loaded_md;
            }
        }
    }
    
    std::cout << "Warming up for 5 iterations..." << std::endl;
    
    // Warm-up iterations
    for (int iter = 0; iter < 5; iter++) {
        std::vector<nixlXferReqH*> req_handles(num_gpus);
        
        // Create ring transfer pattern: GPU i sends to GPU (i+1)
        for (int i = 0; i < num_gpus; i++) {
            int src = i;
            int dst = (i + 1) % num_gpus;
            std::string dst_agent = "Agent" + std::to_string(dst);
            
            // Source descriptors (from src GPU)
            nixl_xfer_dlist_t src_descs(VRAM_SEG);
            nixlBasicDesc src_desc;
            src_desc.addr = reinterpret_cast<uintptr_t>(d_cache[src]);
            src_desc.len = cache_size;
            src_desc.devId = src;
            src_descs.addDesc(src_desc);
            
            // Destination descriptors (to dst GPU)
            nixl_xfer_dlist_t dst_descs(VRAM_SEG);
            nixlBasicDesc dst_desc;
            dst_desc.addr = reinterpret_cast<uintptr_t>(d_cache[dst]);
            dst_desc.len = cache_size;
            dst_desc.devId = dst;
            dst_descs.addDesc(dst_desc);
            
            // Create transfer request
            nixlCheckError(agents[src]->createXferReq(NIXL_WRITE, src_descs, dst_descs,
                                                     dst_agent, req_handles[i], &extra_params[src]),
                          "Failed to create transfer request");
            
            // Post transfer request
            nixlCheckError(agents[src]->postXferReq(req_handles[i], &extra_params[src]),
                          "Failed to post transfer request");
        }
        
        // Wait for all transfers to complete
        // NIXL_IN_PROG (1) = in progress, NIXL_SUCCESS (0) = complete, negative = error
        bool all_done = false;
        while (!all_done) {
            all_done = true;
            for (int i = 0; i < num_gpus; i++) {
                nixl_status_t status = agents[i]->getXferStatus(req_handles[i]);
                if (status < NIXL_SUCCESS) {
                    // Error occurred
                    fprintf(stderr, "ERROR: Transfer failed with status %d\n", status);
                    exit(1);
                }
                if (status != NIXL_SUCCESS) {
                    // Still in progress (NIXL_IN_PROG)
                    all_done = false;
                }
            }
        }
        
        // Release transfer requests
        for (int i = 0; i < num_gpus; i++) {
            nixlCheckError(agents[i]->releaseXferReq(req_handles[i]),
                          "Failed to release transfer request");
        }
    }
    
    std::cout << "Benchmarking for " << iterations << " iterations..." << std::endl;
    
    // Benchmark iterations
    std::vector<double> latencies;
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        // Record start time
        cudaCheckError(::cudaSetDevice(0));
        cudaCheckError(::cudaEventRecord(start_events[0], streams[0]));
        
        std::vector<nixlXferReqH*> req_handles(num_gpus);
        
        // Create ring transfer pattern
        for (int i = 0; i < num_gpus; i++) {
            int src = i;
            int dst = (i + 1) % num_gpus;
            std::string dst_agent = "Agent" + std::to_string(dst);
            
            // Source descriptors
            nixl_xfer_dlist_t src_descs(VRAM_SEG);
            nixlBasicDesc src_desc;
            src_desc.addr = reinterpret_cast<uintptr_t>(d_cache[src]);
            src_desc.len = cache_size;
            src_desc.devId = src;
            src_descs.addDesc(src_desc);
            
            // Destination descriptors
            nixl_xfer_dlist_t dst_descs(VRAM_SEG);
            nixlBasicDesc dst_desc;
            dst_desc.addr = reinterpret_cast<uintptr_t>(d_cache[dst]);
            dst_desc.len = cache_size;
            dst_desc.devId = dst;
            dst_descs.addDesc(dst_desc);
            
            // Create and post transfer request
            nixlCheckError(agents[src]->createXferReq(NIXL_WRITE, src_descs, dst_descs,
                                                     dst_agent, req_handles[i], &extra_params[src]),
                          "Failed to create transfer request");
            nixlCheckError(agents[src]->postXferReq(req_handles[i], &extra_params[src]),
                          "Failed to post transfer request");
        }
        
        // Wait for all transfers to complete
        // NIXL_IN_PROG (1) = in progress, NIXL_SUCCESS (0) = complete, negative = error
        bool all_done = false;
        while (!all_done) {
            all_done = true;
            for (int i = 0; i < num_gpus; i++) {
                nixl_status_t status = agents[i]->getXferStatus(req_handles[i]);
                if (status < NIXL_SUCCESS) {
                    // Error occurred
                    fprintf(stderr, "ERROR: Transfer failed with status %d\n", status);
                    exit(1);
                }
                if (status != NIXL_SUCCESS) {
                    // Still in progress (NIXL_IN_PROG)
                    all_done = false;
                }
            }
        }
        
        // Record end time
        cudaCheckError(::cudaSetDevice(0));
        cudaCheckError(::cudaEventRecord(end_events[0], streams[0]));
        cudaCheckError(::cudaEventSynchronize(end_events[0]));
        
        // Calculate latency
        float elapsed_ms;
        cudaCheckError(::cudaEventElapsedTime(&elapsed_ms, start_events[0], end_events[0]));
        latencies.push_back(elapsed_ms);
        
        // Release transfer requests
        for (int i = 0; i < num_gpus; i++) {
            nixlCheckError(agents[i]->releaseXferReq(req_handles[i]),
                          "Failed to release transfer request");
        }
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();
    
    // Calculate statistics
    double sum = 0.0, min_lat = latencies[0], max_lat = latencies[0];
    for (double lat : latencies) {
        sum += lat;
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
    }
    double avg_latency = sum / latencies.size();
    
    // Calculate throughput (GB/s)
    double data_transferred_gb = (cache_size * iterations * num_gpus) / (1024.0 * 1024.0 * 1024.0);
    double throughput_gbps = data_transferred_gb / (total_time_ms / 1000.0);
    
    // Cleanup
    std::cout << "Cleaning up NIXL resources..." << std::endl;
    
    for (int i = 0; i < num_gpus; i++) {
        // Invalidate remote metadata
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                std::string agent_name = "Agent" + std::to_string(j);
                agents[i]->invalidateRemoteMD(agent_name);
            }
        }
        
        // Deregister memory
        nixlCheckError(agents[i]->deregisterMem(*reg_lists[i], &extra_params[i]),
                      "Failed to deregister memory");
        
        // Delete registration list
        delete reg_lists[i];
        
        // Delete agent
        delete agents[i];
        
        // Cleanup CUDA resources
        cudaCheckError(::cudaSetDevice(i));
        cudaCheckError(::cudaFree(d_cache[i]));
        cudaCheckError(::cudaStreamDestroy(streams[i]));
        cudaCheckError(::cudaEventDestroy(start_events[i]));
        cudaCheckError(::cudaEventDestroy(end_events[i]));
    }
    
    delete[] d_cache;
    delete[] streams;
    delete[] start_events;
    delete[] end_events;
    
    return {avg_latency, throughput_gbps, min_lat, max_lat};
}
#endif

/**
 * @brief Baseline benchmark using cudaMemcpy for comparison
 */
BenchmarkResult benchmarkCudaMemcpy(const KVCacheConfig& config, int num_gpus, int iterations) {
    std::cout << "\n=== Baseline cudaMemcpy Benchmark ===" << std::endl;
    
    size_t cache_size = config.get_total_cache_size();
    
    float** d_cache = new float*[num_gpus];
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    
    for (int i = 0; i < num_gpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        cudaCheckError(::cudaMalloc(&d_cache[i], cache_size));
        cudaCheckError(::cudaStreamCreate(&streams[i]));
        cudaCheckError(::cudaMemset(d_cache[i], 0, cache_size));
    }
    
    // Enable peer access
    for (int i = 0; i < num_gpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access;
                cudaCheckError(::cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t err = ::cudaDeviceEnablePeerAccess(j, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        cudaCheckError(err);
                    }
                }
            }
        }
    }
    
    // Warm-up
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 0; i < num_gpus; i++) {
            int src = i;
            int dst = (i + 1) % num_gpus;
            cudaCheckError(::cudaSetDevice(src));
            cudaCheckError(::cudaMemcpyAsync(d_cache[dst], d_cache[src], cache_size,
                                            cudaMemcpyDeviceToDevice, streams[src]));
        }
        for (int i = 0; i < num_gpus; i++) {
            cudaCheckError(::cudaSetDevice(i));
            cudaCheckError(::cudaStreamSynchronize(streams[i]));
        }
    }
    
    // Benchmark
    std::vector<double> latencies;
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_gpus; i++) {
            int src = i;
            int dst = (i + 1) % num_gpus;
            cudaCheckError(::cudaSetDevice(src));
            cudaCheckError(::cudaMemcpyAsync(d_cache[dst], d_cache[src], cache_size,
                                            cudaMemcpyDeviceToDevice, streams[src]));
        }
        
        for (int i = 0; i < num_gpus; i++) {
            cudaCheckError(::cudaSetDevice(i));
            cudaCheckError(::cudaStreamSynchronize(streams[i]));
        }
        
        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_time_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
        latencies.push_back(iter_time_ms);
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();
    
    // Calculate statistics
    double sum = 0.0, min_lat = latencies[0], max_lat = latencies[0];
    for (double lat : latencies) {
        sum += lat;
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
    }
    double avg_latency = sum / latencies.size();
    
    // Calculate throughput (GB/s)
    double data_transferred_gb = (cache_size * iterations * num_gpus) / (1024.0 * 1024.0 * 1024.0);
    double throughput_gbps = data_transferred_gb / (total_time_ms / 1000.0);
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaCheckError(::cudaSetDevice(i));
        cudaCheckError(::cudaFree(d_cache[i]));
        cudaCheckError(::cudaStreamDestroy(streams[i]));
    }
    
    delete[] d_cache;
    delete[] streams;
    
    return {avg_latency, throughput_gbps, min_lat, max_lat};
}

void printResults(const std::string& method, const BenchmarkResult& result, size_t cache_size_mb) {
    std::cout << "\n" << method << " Results:" << std::endl;
    std::cout << "  Average Latency: " << std::fixed << std::setprecision(3) 
              << result.avg_latency_ms << " ms" << std::endl;
    std::cout << "  Min Latency:     " << result.min_latency_ms << " ms" << std::endl;
    std::cout << "  Max Latency:     " << result.max_latency_ms << " ms" << std::endl;
    std::cout << "  Throughput:      " << std::setprecision(2) 
              << result.throughput_gbps << " GB/s" << std::endl;
    std::cout << "  Cache Size:      " << cache_size_mb << " MB" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "NCCL vs NIXL: KV Cache Transfer Benchmark" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "\nBased on: https://uccl-project.github.io/posts/kv-transfer-engine/" << std::endl;
    
    // Check available GPUs
    int num_gpus = 0;
    cudaCheckError(::cudaGetDeviceCount(&num_gpus));
    
    if (num_gpus < 2) {
        std::cerr << "Error: This benchmark requires at least 2 GPUs. Found: " 
                  << num_gpus << std::endl;
        std::cerr << "Note: You can simulate multi-GPU behavior on a single GPU for testing." << std::endl;
        return 1;
    }
    
    std::cout << "\nDetected " << num_gpus << " GPU(s)" << std::endl;
    
    // Print GPU information
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaCheckError(::cudaGetDeviceProperties(&prop, i));
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    }
    
    // Configure KV cache (typical LLaMA-2 70B configuration)
    KVCacheConfig config;
    config.num_layers = 80;
    config.num_heads = 64;
    config.head_dim = 128;
    config.seq_len = 2048;
    config.batch_size = 1;
    
    size_t cache_size = config.get_total_cache_size();
    size_t cache_size_mb = cache_size / (1024 * 1024);
    
    std::cout << "\nKV Cache Configuration:" << std::endl;
    std::cout << "  Layers:      " << config.num_layers << std::endl;
    std::cout << "  Heads:       " << config.num_heads << std::endl;
    std::cout << "  Head Dim:    " << config.head_dim << std::endl;
    std::cout << "  Seq Length:  " << config.seq_len << std::endl;
    std::cout << "  Batch Size:  " << config.batch_size << std::endl;
    std::cout << "  Cache Size:  " << cache_size_mb << " MB per GPU" << std::endl;
    
    int iterations = 100;
    std::cout << "\nRunning " << iterations << " iterations per method..." << std::endl;
    
    // Run benchmarks
    BenchmarkResult baseline = benchmarkCudaMemcpy(config, num_gpus, iterations);
    printResults("Baseline cudaMemcpy", baseline, cache_size_mb);
    
#ifdef USE_NCCL
    BenchmarkResult nccl_result = benchmarkNCCL(config, num_gpus, iterations);
    printResults("NCCL", nccl_result, cache_size_mb);
#else
    std::cout << "\n[NCCL not available - compile with -DUSE_NCCL and link with NCCL library]" << std::endl;
#endif
    
#ifdef USE_NIXL
    BenchmarkResult nixl_result = benchmarkNIXL(config, num_gpus, iterations);
    printResults("NIXL", nixl_result, cache_size_mb);
#else
    std::cout << "\n[NIXL not available - compile with -DUSE_NIXL and link with NIXL library]" << std::endl;
#endif
    
    // Summary
    std::cout << "\n============================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "\nKey Differences:" << std::endl;
    std::cout << "1. NCCL: Uses GPU SM resources for P2P transfers" << std::endl;
    std::cout << "   - Can interfere with inference computation" << std::endl;
    std::cout << "   - Suitable for training workloads" << std::endl;
    std::cout << "\n2. NIXL: Offloads transfers from GPU" << std::endl;
    std::cout << "   - Frees SM resources for computation" << std::endl;
    std::cout << "   - Optimized for inference workloads" << std::endl;
    std::cout << "   - Lower latency for typical KV cache sizes" << std::endl;
    std::cout << "\n3. cudaMemcpy: Basic P2P transfer" << std::endl;
    std::cout << "   - Baseline comparison" << std::endl;
    std::cout << "   - Limited scalability" << std::endl;
    
    std::cout << "\n============================================" << std::endl;
    
    return 0;
}

