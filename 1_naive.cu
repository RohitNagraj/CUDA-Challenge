#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 27)
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BASELINE 256
#define DELTA 0.0001

__global__ void transform(float *arr, float *output, double *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        if ((i % 4 == 0) && (i % 32 < 16))
            output[i] = sinf(arr[i]);
        else if ((i % 4 == 1) && (i % 32 < 16))
            output[i] = cosf(arr[i]);
        else if ((i % 4 == 2) && (i % 32 < 16))
            output[i] = logf(arr[i]);
        else if ((i % 4 == 3) && (i % 32 < 16))
            output[i] = expf(arr[i]);
        else if ((i % 4 == 0) && (i % 32 >= 16))
            output[i] = sinf(arr[i]) * sinf(arr[i - 16]);
        else if ((i % 4 == 1) && (i % 32 >= 16))
            output[i] = cosf(arr[i]) * cosf(arr[i - 16]);
        else if ((i % 4 == 2) && (i % 32 >= 16))
            output[i] = logf(arr[i]) * logf(arr[i - 16]);
        else if ((i % 4 == 3) && (i % 32 >= 16))
            output[i] = expf(arr[i]) * expf(arr[i - 16]);

        __syncthreads();
        if ((i % 4 == 1) && (output[i] > 0.5))
            atomicAdd(sum, (double)output[i - 1]);
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
        h_arr[i] = (float)rand() / RAND_MAX * 5.0f;

    dim3 grid(N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);
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
    printf("Sum - Actual: %f, Expected: 566300.125\n", (float)h_sum);

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