#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// #define N 100'000'000
#define N 100'000'000
#define BLOCK_SIZE 32
#define N_STREAMS 1

__global__ void transform(float *arr, float *output, float *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        if ((i % 4 == 0) && (i % 32 < 16))
        {
            output[i] = sinf(arr[i]);
        }
        else if ((i % 4 == 1) && (i % 32 < 16))
        {
            {
                output[i] = cosf(arr[i]);
                if (output[i] > 0.5)
                    atomicAdd(sum, output[i]);
            }
        }
        else if ((i % 4 == 2) && (i % 32 < 16))
        {
            output[i] = logf(arr[i]);
        }
        else if ((i % 4 == 3) && (i % 32 < 16))
        {
            output[i] = expf(arr[i]);
        }
        else if ((i % 4 == 0) && (i % 32 >= 16))
        {
            output[i] = sinf(arr[i]) * sinf(arr[i - 16]);
        }
        else if ((i % 4 == 1) && (i % 32 >= 16))
        {
            output[i] = cosf(arr[i]) * cosf(arr[i - 16]);
        }
        else if ((i % 4 == 2) && (i % 32 >= 16))
        {
            output[i] = logf(arr[i]) * logf(arr[i - 16]);
        }
        else if ((i % 4 == 3) && (i % 32 >= 16))
        {
            output[i] = expf(arr[i]) * expf(arr[i - 16]);
        }
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
    float *h_arr, *h_output, h_sum = 0.0;
    float *d_arr, *d_output, *d_sum;
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
    cudaMalloc(&d_sum, sizeof(float));

    for (i = 0; i < N; i++)
    {
        h_arr[i] = (float)rand() / RAND_MAX * 5.0f;
    }

    dim3 grid(N / (BLOCK_SIZE * N_STREAMS));
    dim3 block(BLOCK_SIZE);

    // Host to Device Memcpy
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up
    baseline<<<grid, block>>>(d_arr, d_output);
    cudaDeviceSynchronize();

    cudaEventRecord(startBaseline, 0);
    baseline<<<grid, block>>>(d_arr, d_output);
    cudaEventRecord(stopBaseline, 0);
    cudaEventSynchronize(stopBaseline);

    cudaEventRecord(start, 0);
    transform<<<grid, block>>>(d_arr, d_output, d_sum);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&elapsedTimeBaseline, startBaseline, stopBaseline);
    printf("Baseline execution time: %.3f ms\n", elapsedTimeBaseline);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %.3f ms\n", elapsedTime);

    // printf("First 32 inputs and transformations:\n");
    // for (int i = 0; i < 32; i++)
    //     printf("Input: %f, Transformation: %f\n", h_arr[i], h_output[i]);
    printf("Sum of Sin: %f\n", h_sum);

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