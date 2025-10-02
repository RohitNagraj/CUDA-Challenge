#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define N (1 << 27)
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BASELINE 256
#define DELTA 0.01

__global__ void transform(__nv_bfloat16 *arr, __nv_bfloat16 *output, float *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 4; // Each thread processes 4 elements
    
    float thread_sum = 0.0f;
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    __shared__ __nv_bfloat16 cache[BLOCK_SIZE * 4 + 16];

    // Load previous 16 elements into shared memory (with bounds checking)
    if (threadIdx.x < 4) {
        int offset = blockIdx.x * BLOCK_SIZE * 4 - 16 + threadIdx.x * 4;
        if (offset >= 0 && offset + 3 < N) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                cache[threadIdx.x * 4 + k] = arr[offset + k];
            }
        }
    }

    // Load main data into shared memory
    if (i + 3 < N) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            cache[16 + threadIdx.x * 4 + k] = arr[i + k];
        }
    }
    
    __syncthreads();

    if (i + 3 < N)
    {
        __nv_bfloat16 output_vec[4];
        
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int idx = i + j;
            float xi = __bfloat162float(cache[16 + threadIdx.x * 4 + j]);
            int mod4 = (idx & 3);
            bool first_block = ((idx & 31) < 16);
            
            // Compute base transformation using fast math
            float result;
            if (mod4 == 0)
                result = __sinf(xi);
            else if (mod4 == 1)
                result = __cosf(xi);
            else if (mod4 == 2)
                result = __logf(xi);
            else
                result = __expf(xi);
            
            // Apply multiplication for second half of 32-element blocks
            if (!first_block)
            {
                float ximinus16 = __bfloat162float(cache[threadIdx.x * 4 + j]);
                float prev_result;
                
                if (mod4 == 0)
                    prev_result = __sinf(ximinus16);
                else if (mod4 == 1)
                    prev_result = __cosf(ximinus16);
                else if (mod4 == 2)
                    prev_result = __logf(ximinus16);
                else
                    prev_result = __expf(ximinus16);
                
                result *= prev_result;
            }
            
            output_vec[j] = __float2bfloat16(result);
        }
        
        // Store 4 bfloat16s
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            output[i + k] = output_vec[k];
        }
        
        // Now accumulate for sum (when idx % 4 == 1 and result > 0.5)
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int idx = i + j;
            float result_f = __bfloat162float(output_vec[j]);
            if ((idx % 4 == 1) && (result_f > 0.5f))
            {
                if (j > 0) {
                    thread_sum += __bfloat162float(output_vec[j - 1]);
                } else if (idx > 0) {
                    thread_sum += __bfloat162float(output[idx - 1]);
                }
            }
        }
    }

    // Reduce within warp using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    // First thread of each warp writes to shared memory
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0)
        warp_sums[warp_id] = thread_sum;

    __syncthreads();

    // First warp reduces the warp sums
    if (threadIdx.x == 0)
    {
        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE / 32; j++)
            block_sum += warp_sums[j];
        
        if (block_sum != 0.0f)
            atomicAdd(sum, block_sum);
    }
}

__global__ void baseline(__nv_bfloat16 *arr, __nv_bfloat16 *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = arr[i];
}

int main()
{
    srand(12345);
    float *h_arr_float;
    __nv_bfloat16 *h_arr, *h_output;
    float h_sum = 0.0f;
    __nv_bfloat16 *d_arr, *d_output;
    float *d_sum;
    int i;
    float elapsedTimeBaseline, elapsedTime;

    cudaEvent_t startBaseline, stopBaseline, start, stop;

    cudaEventCreate(&startBaseline);
    cudaEventCreate(&start);
    cudaEventCreate(&stopBaseline);
    cudaEventCreate(&stop);

    // Allocate float arrays for initialization and verification
    h_arr_float = (float*)malloc(N * sizeof(float));

    cudaMallocHost(&h_arr, N * sizeof(__nv_bfloat16));
    cudaMallocHost(&h_output, N * sizeof(__nv_bfloat16));

    cudaMalloc(&d_arr, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_sum, sizeof(float));

    // Initialize with float, then convert to bfloat16
    for (i = 0; i < N; i++)
    {
        h_arr_float[i] = (float)rand() / RAND_MAX * 5.0f;
        h_arr[i] = __float2bfloat16(h_arr_float[i]);
    }

    // Each thread processes 4 elements, so divide grid size by 4
    dim3 grid(N / (BLOCK_SIZE * 4));
    dim3 block(BLOCK_SIZE);

    // Host to Device Memcpy
    cudaMemcpy(d_arr, h_arr, N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up
    baseline<<<N / BLOCK_SIZE_BASELINE, BLOCK_SIZE_BASELINE>>>(d_arr, d_output);
    cudaDeviceSynchronize();

    cudaEventRecord(startBaseline, 0);
    baseline<<<N / BLOCK_SIZE_BASELINE, BLOCK_SIZE_BASELINE>>>(d_arr, d_output);
    cudaEventRecord(stopBaseline, 0);
    cudaEventSynchronize(stopBaseline);

    cudaEventRecord(start, 0);
    transform<<<grid, block>>>(d_arr, d_output, d_sum);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_output, d_output, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Correctness Testing
    bool correct = true;

    for (int i = 0; i < 64; i++)
    {
        if ((i % 4 == 0) && (i % 32 < 16))
            if (fabsf((float)h_output[i] - sinf((float)h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", sinf((float)h_arr[i]), (float)h_output[i], i);
            }
        else if ((i % 4 == 1) && (i % 32 < 16))
            if (fabsf((float)h_output[i] - cosf((float)h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", cosf((float)h_arr[i]), (float)h_output[i], i);
            }
        else if ((i % 4 == 2) && (i % 32 < 16))
            if (fabsf((float)h_output[i] - logf((float)h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", logf((float)h_arr[i]), (float)h_output[i], i);
            }
        else if ((i % 4 == 3) && (i % 32 < 16))
            if (fabsf((float)h_output[i] - expf((float)h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", expf((float)h_arr[i]), (float)h_output[i], i);
            }
        else if ((i % 4 == 0) && (i % 32 >= 16))
            if (fabsf((float)h_output[i] - (sinf((float)h_arr[i]) * sinf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (sinf((float)h_arr[i]) * sinf(h_arr[i - 16])), (float)h_output[i], i);
            }
        else if ((i % 4 == 1) && (i % 32 >= 16))
            if (fabsf((float)h_output[i] - (cosf((float)h_arr[i]) * cosf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (cosf((float)h_arr[i]) * cosf(h_arr[i - 16])), (float)h_output[i], i);
            }
        else if ((i % 4 == 2) && (i % 32 >= 16))
            if (fabsf((float)h_output[i] - (logf((float)h_arr[i]) * logf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (logf((float)h_arr[i]) * logf(h_arr[i - 16])), (float)h_output[i], i);
            }
        else if ((i % 4 == 3) && (i % 32 >= 16))
            if (fabsf((float)h_output[i] - (expf((float)h_arr[i]) * expf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (expf((float)h_arr[i]) * expf(h_arr[i - 16])), (float)h_output[i], i);
            }
    }
    if (!correct)
        printf("CORRECTNESS TEST FAILED!\n");
    else
        printf("Correctness Tests Passed!\n");
    printf("Sum - Actual: %f, Expected: 566300.125\n", (float)h_sum);

    cudaEventElapsedTime(&elapsedTimeBaseline, startBaseline, stopBaseline);
    printf("Baseline execution time: %.3f ms\n", elapsedTimeBaseline);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %.3f ms\n", elapsedTime);

    printf("Speed: %f%%\n", (elapsedTimeBaseline / elapsedTime) * 100);

    double bytes_baseline = 2.0 * N * sizeof(__nv_bfloat16);
    double bandwidth_baseline = (bytes_baseline / 1e9) / (elapsedTimeBaseline / 1000.0);
    printf("Baseline Bandwidth: %.2f GB/s\n", bandwidth_baseline);

    double bytes_transform = 2.5 * N * sizeof(__nv_bfloat16);
    double bandwidth_transform = (bytes_transform / 1e9) / (elapsedTime / 1000.0);
    printf("Kernel Bandwidth: %.2f GB/s\n", bandwidth_transform);

    cudaFreeHost(h_arr);
    cudaFreeHost(h_output);

    cudaFree(d_arr);
    cudaFree(d_output);
    cudaFree(d_sum);

    cudaEventDestroy(startBaseline);
    cudaEventDestroy(stopBaseline);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}