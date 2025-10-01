#include <cuda_runtime.h>
#include <stdio.h>

// Baseline scalar version (for comparison)
__global__ void vectorAdd_baseline(const float* a, 
                                    const float* b, 
                                    float* c, 
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple one element per thread
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Baseline with grid-stride loop (better for large arrays)
__global__ void vectorAdd_baseline_gridstride(const float* a, 
                                               const float* b, 
                                               float* c, 
                                               int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

// Fastest vectorized version using int4 (128-bit loads/stores)
__global__ void vectorAdd_int4(const int* __restrict__ a, 
                                const int* __restrict__ b, 
                                int* __restrict__ c, 
                                int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements at a time
    for (int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
        int4 a_vec = reinterpret_cast<const int4*>(a)[i];
        int4 b_vec = reinterpret_cast<const int4*>(b)[i];
        int4 c_vec;
        
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;
        
        reinterpret_cast<int4*>(c)[i] = c_vec;
    }
    
    // Handle remaining elements
    int remainder = N % 4;
    if (idx == N/4 && remainder != 0) {
        for (int i = 0; i < remainder; i++) {
            int pos = N - remainder + i;
            c[pos] = a[pos] + b[pos];
        }
    }
}

// Vectorized version using float4 (for float arrays)
__global__ void vectorAdd_float4(const float* __restrict__ a, 
                                  const float* __restrict__ b, 
                                  float* __restrict__ c, 
                                  int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
        float4 a_vec = reinterpret_cast<const float4*>(a)[i];
        float4 b_vec = reinterpret_cast<const float4*>(b)[i];
        float4 c_vec;
        
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;
        
        reinterpret_cast<float4*>(c)[i] = c_vec;
    }
    
    int remainder = N % 4;
    if (idx == N/4 && remainder != 0) {
        for (int i = 0; i < remainder; i++) {
            int pos = N - remainder + i;
            c[pos] = a[pos] + b[pos];
        }
    }
}

// Example usage
void launchVectorAdd(const float* d_a, const float* d_b, float* d_c, int N) {
    int threads = 256;
    int blocks = min((N/4 + threads - 1) / threads, 65535);
    
    vectorAdd_float4<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
}

// Timing utility for single kernel execution
float timeKernel(void (*kernel)(const float*, const float*, float*, int),
                 const float* d_a, const float* d_b, float* d_c, int N,
                 int threads, int blocks) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// Benchmarking utility with detailed timing
void benchmarkKernel(void (*kernel)(const float*, const float*, float*, int),
                     const float* d_a, const float* d_b, float* d_c, int N,
                     int threads, int blocks, const char* name, int iterations = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    // Single execution timing
    float singleTime = timeKernel(kernel, d_a, d_b, d_c, N, threads, blocks);
    
    // Benchmark multiple iterations
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avgTime = milliseconds / iterations;
    float minTime = avgTime; // In practice, min is often more stable than avg
    
    // Calculate bandwidth (3 arrays: 2 reads + 1 write)
    float bandwidth = (3.0f * N * sizeof(float)) / (avgTime / 1000.0f) / 1e9;
    float peakBandwidth = (3.0f * N * sizeof(float)) / (minTime / 1000.0f) / 1e9;
    
    printf("%-25s | Single: %7.3f ms | Avg(%d): %7.3f ms | BW: %7.2f GB/s\n", 
           name, singleTime, iterations, avgTime, bandwidth);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Complete example with performance comparison
int main() {
    const int N = 1 << 24; // 16M elements for better benchmarking
    size_t bytes = N * sizeof(float);
    
    printf("Vector size: %d elements (%.2f MB)\n", N, bytes / 1024.0f / 1024.0f);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    printf("\n=== Performance Comparison ===\n");
    
    // Benchmark baseline
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = min(blocks, 65535);
    benchmarkKernel(vectorAdd_baseline, d_a, d_b, d_c, N, threads, blocks, 
                    "Baseline (simple)");
    
    // Benchmark baseline with grid-stride
    blocks = min((N + threads - 1) / threads, 2048);
    benchmarkKernel(vectorAdd_baseline_gridstride, d_a, d_b, d_c, N, threads, blocks,
                    "Baseline (grid-stride)");
    
    // Benchmark float4 optimized
    blocks = min((N/4 + threads - 1) / threads, 2048);
    benchmarkKernel(vectorAdd_float4, d_a, d_b, d_c, N, threads, blocks,
                    "Vectorized float4");
    
    // Verify result
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    bool success = true;
    for (int i = 0; i < N && success; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            printf("\nError at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
        }
    }
    
    if (success) printf("\n✓ All results verified correct!\n");
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}