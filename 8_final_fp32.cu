#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 27)
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BASELINE 256
#define DELTA 0.0001

__global__ void transform(float *arr, float *output, double *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 4; // Each thread processes 4 elements

    double thread_sum = 0.0;
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    __shared__ float cache[BLOCK_SIZE * 4];

    // Load main data into shared memory
    if (i + 3 < N)
        *reinterpret_cast<float4 *>(&cache[threadIdx.x * 4]) = *reinterpret_cast<float4 *>(&arr[i]);

    __syncthreads();

    if (i + 3 < N)
    {
        float4 output_vec;
        bool first_block = ((i & 31) < 16);

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float xi = cache[threadIdx.x * 4 + j];
            int mod4 = j;

            float result;
            if (mod4 == 0)
                result = __sinf(xi);
            else if (mod4 == 1)
                result = __cosf(xi);
            else if (mod4 == 2)
                result = __logf(xi);
            else
                result = __expf(xi);

            if (!first_block)
            {
                float ximinus16 = cache[threadIdx.x * 4 + j - 16];
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

            ((float *)&output_vec)[j] = result;
        }

        // Store 4 floats at once using float4
        *reinterpret_cast<float4 *>(&output[i]) = output_vec;

        // Now accumulate for sum (when idx % 4 == 1 and result > 0.5)
        if (((float *)&output_vec)[1] > 0.5f)
            thread_sum = (double)((float *)&output_vec)[0];
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

    // First thread reduces the warp sums
    if (threadIdx.x == 0)
    {
        double block_sum = 0.0;
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE / 32; j++)
            block_sum += warp_sums[j];

        if (block_sum != 0.0)
            atomicAdd(sum, block_sum);
    }
}

__global__ void baseline(float *arr, float *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = arr[i];
}

int main()
{
    srand(12345);
    float *h_arr, *h_output;
    double h_sum = 0.0;
    float *d_arr, *d_output;
    double *d_sum;
    int i;
    float elapsedTimeBaseline, elapsedTime;

    cudaEvent_t startBaseline, stopBaseline, start, stop;

    cudaEventCreate(&startBaseline);
    cudaEventCreate(&start);
    cudaEventCreate(&stopBaseline);
    cudaEventCreate(&stop);

    cudaMallocHost(&h_arr, N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));

    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(double));

    for (i = 0; i < N; i++)
    {
        h_arr[i] = (float)rand() / RAND_MAX * 5.0f;
    }

    // Each thread processes 4 elements, so divide grid size by 4
    dim3 grid(N / (BLOCK_SIZE * 4)); // 2 ^ 27 / (1024) = 128k
    dim3 block(BLOCK_SIZE);          // 256

    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

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

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Correctness Testing
    bool correct = true;

    for (int i = 0; i < 64; i++)
    {
        if ((i % 4 == 0) && (i % 32 < 16))
            if (fabsf(h_output[i] - sinf(h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", sinf(h_arr[i]), h_output[i], i);
            }
        else if ((i % 4 == 1) && (i % 32 < 16))
            if (fabsf(h_output[i] - cosf(h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", cosf(h_arr[i]), h_output[i], i);
            }
        else if ((i % 4 == 2) && (i % 32 < 16))
            if (fabsf(h_output[i] - logf(h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", logf(h_arr[i]), h_output[i], i);
            }
        else if ((i % 4 == 3) && (i % 32 < 16))
            if (fabsf(h_output[i] - expf(h_arr[i])) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", expf(h_arr[i]), h_output[i], i);
            }
        else if ((i % 4 == 0) && (i % 32 >= 16))
            if (fabsf(h_output[i] - (sinf(h_arr[i]) * sinf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (sinf(h_arr[i]) * sinf(h_arr[i - 16])), h_output[i], i);
            }
        else if ((i % 4 == 1) && (i % 32 >= 16))
            if (fabsf(h_output[i] - (cosf(h_arr[i]) * cosf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (cosf(h_arr[i]) * cosf(h_arr[i - 16])), h_output[i], i);
            }
        else if ((i % 4 == 2) && (i % 32 >= 16))
            if (fabsf(h_output[i] - (logf(h_arr[i]) * logf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (logf(h_arr[i]) * logf(h_arr[i - 16])), h_output[i], i);
            }
        else if ((i % 4 == 3) && (i % 32 >= 16))
            if (fabsf(h_output[i] - (expf(h_arr[i]) * expf(h_arr[i - 16]))) > DELTA)
            {
                correct = false;
                printf("Expected: %f, Got: %f at index: %d\n", (expf(h_arr[i]) * expf(h_arr[i - 16])), h_output[i], i);
            }
    }
    if (!correct)
        printf("CORRECTNESS TEST FAILED!\n");
    else
        printf("Correctness Tests Passed!\n");
    printf("Sum - Actual: %f, Expected: 566300.25\n", (float)h_sum);

    cudaEventElapsedTime(&elapsedTimeBaseline, startBaseline, stopBaseline);
    printf("Baseline execution time: %.3f ms\n", elapsedTimeBaseline);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %.3f ms\n", elapsedTime);

    printf("Speed: %f%%\n", (elapsedTimeBaseline / elapsedTime) * 100);

    double bytes_baseline = 2.0 * N * sizeof(float);
    double bandwidth_baseline = (bytes_baseline / 1e9) / (elapsedTimeBaseline / 1000.0);
    printf("Baseline Bandwidth: %.2f GB/s\n", bandwidth_baseline);

    double bytes_transform = 2.5 * N * sizeof(float);
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