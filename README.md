# CUDA-Challenge

## Final Submission
**Github URL**: https://github.com/RohitNagraj/CUDA-Challenge  
**FP32**: [8.5_final_fp32_opt.cu](https://github.com/RohitNagraj/CUDA-Challenge/blob/main/8.5_final_fp32_opt.cu)  
**BF16**: [9_bf16.cu](https://github.com/RohitNagraj/CUDA-Challenge/blob/main/9_bf16.cu)  
| Program | Baseline Execution Time | Kernel Execution Time | Speedup | Baseline Bandwidth | Kernel Bandwidth | DType |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 8.5_final_fp32_opt | 0.446ms | 0.376ms | 118.30% | 2404.82 GB/s | 3556.30 GB/s * | FP32 |
| 9_bf16 | 0.372ms | 0.287ms | 129.47% | 1443.95 GB/s | 2336.92 GB/s * | BF16 |

\* -> Note that these bandwidths are not purely for HBM, since we cache the `i-16` blocks. This bandwidth is higher than the theoretical HBM bandwidth for H100 because part of it comes from the shared memory.

Detailed Results in the **Solution** section below.
## Problem Statement
We want to perform the following transformation on a large 1-D array of floating point data:
1. As follows:  
a. `x_i -> sin(x_i)` if `i % 4 == 0 && i % 32 < 16`  
b. `x_i -> cos(x_i)` if `i % 4 == 1 && i % 32 < 16`  
c. `x_i -> log(x_i)` if `i % 4 == 2 && i % 32 < 16`  
d. `x_i -> exp(x_i)` if `i % 4 == 3 && i % 32 < 16`  
e. `x_i -> sin(x_[i-16]) * sin(x_i)` if `i % 4 == 0 && i % 32 >= 16`  
f. `x_i -> cos(x_[i-16]) * cos(x_i)` if `i % 4 == 1 && i % 32 >= 16`  
g. `x_i -> log(x_[i-16]) * log(x_i)` if `i % 4 == 2 && i % 32 >= 16`  
h. `x_i -> exp(x_[i-16]) * exp(x_i)` if `i % 4 == 3 && i % 32 >= 16`  
2. The kernel should also calculate the global sum of the sin terms wherever thecorresponding cos term value is greater than 0.5. In the ideal solution, the value of this sum would be identical from run to run.


The array should consist of 100M random fp32 numbers between (0, 5). The goal for the
transformation is that the achieved bandwidth (bytes read + bytes written) / time should be as
close as possible to the speed of a memcpy. Report the bandwidth achieved for a memcpy
(cudaMemcpy(DevicetoDevice)) and the bandwidth achieved for the kernel. Measure only time on the device, time spent copying data to/from the device is not important. Include timing in the code, do not rely on profilers to report times.


For bonus points repeat for the bfloat16 data type (__nv_bfloat16).

## Challenges
1. Warp divergence
2. Atomic add precision -> Solved with reduction in higher precision
3. Memory coalescing -> Uncoalesced (strided) access when accessing `i-16`.
4. AtomicAdd takes a longg time!

## Solution
I iteratively build up the kernel, one optimization at a time. The following are the results for each kernel, followed by a brief description about each kernel.

**Input Size:** 128M (Using 128M instead of 100M to avoid handling edge cases for simplicity)

**Baseline Bandwidth**: `2` x `N` x `sizeof(float)` / `time_taken`  
**Kernel Bandwidth**: `2.5` x `N` x `sizeof(float)` / `time_taken`  
Please note that the kernel bandwidth takes into account an extra `0.5 x N` acccss for the `i % 32 >= 16` terms.

### Hardware Specs
Device: NVIDIA H100  
VRAM Capacity: 80GB HBM3  
Peak Theoretical Bandwidth: 3.35TB/s  

### Results
| Program | Baseline Execution Time | Kernel Execution Time | Speedup | Baseline Bandwidth | Kernel Bandwidth | DType |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1_naive | 0.448ms | 11.908ms | 3.76% | 2397.43 GB/s | 112.71 GB/s | FP32 |
| 2_op_fusion | 0.448ms | 11.989ms | 3.74% | 2395.21 GB/s | 111.95 GB/s | FP32 |
| 3_shmem | 0.448ms | 7.740ms | 5.78% | 2397.77 GB/s | 173.41 GB/s | FP32 |
| 4_dynamic_blocksize | 0.447ms | 1.400ms | 31.94% | 2400.69 GB/s | 958.54 GB/s | FP32 |
| 5_tree_reduction | 0.449ms | 1.428ms | 31.41% | 2393.84 GB/s | 940.07 GB/s | FP32 |
| 6_warp_reduction | 0.449ms | 1.368ms | 32.84% | 2390.26 GB/s | 981.40 GB/s | FP32 |
| 7_vectorize | 0.448ms | 0.559ms | 80.09% | 2399.14 GB/s | 2401.96 GB/s | FP32 |
| 8_final_fp32 | 0.447ms | 0.381ms | 117.46% | 2400.00 GB/s | 3523.74 GB/s * | FP32 |
| 8.5_final_fp32_opt | 0.446ms | 0.376ms | 118.30% | 2404.82 GB/s | 3556.30 GB/s * | FP32 |
| 9_bf16 | 0.372ms | 0.287ms | 129.47% | 1443.95 GB/s | 2336.92 GB/s * | BF16 |

\* -> Note that these bandwidths are not purely for HBM, since we cache the `i-16` blocks. This bandwidth is higher than the theoretical HBM bandwidth for H100 because part of it comes from the shared memory.

### Program Descriptions
**1_naive**: Naive implmentation of the logic with fixed `BLOCK_SIZE=32` for ease of thinking. Used atomicAdd for reduction across all non-zero threads.  
**2_op_fusion**: Tried to fix warp divergence by getting rid of the `if-else` blocks. Didn't help.  
**3_shmem**: Write each block's terms to sum into local memory, then sequentially compute the sum on first thread.  
**4_dynamic_blocksize**: Thus far, the block size was fixed to 32 for simplicity. Extended code to support any block size between 32 and 1024. Emperically found block size = 256 to be fastest. This improves occupancy and produces significant speedup.  
**5_tree_reduction**: Once the values to be summed are written to shared memory, I was using sequential addition to find the block's sum. Changed that to tree-based reduction to reduce loop iterations.  
**6_warp_reduction**: Moved from shared memory based local reduction to warp-level reduction that uses registers for data-transfer between warps instead of shared memory. I still use shared memory for reduction between warps within a block. Again, tried warp-level reduction and tree-based reduction for this part, but it didn't result in any speedup so ended up using sequential summation.  
**7_vectorize**: Since our problem is naturally blocked in nature (can process 32 elements in isolation), and since our memory access is already coalesced, using vectorized memory access adds a huge boost to the performance by making use of the whole bandwidth of the memory bus to reduce the number of cycles required for data access.  
**8_final_fp32**: Finally, we can make some final tweaks which include, 1) Vectorized access for the `i-16` parts of the array. 2) Using faster math intrinsics (eg: `__sinf()` instead of `sinf()`) did not fail the precision requirements while maintaining speed. 3) Added shared memory caching for the `i-16` lookback. 4) Added `#pragma unroll` to enable scheduler optimizations internally.  
**8.5_final_fp32_opt**: Profiled `8_final_fp32` and found 4-way bank conflicts. Fixed it using structure of arrays for caching in SHMEM.
**9_bf16**: Theoretically, I can achieve higher performance on this since BF16 is half the data of FP32. But for simplicity, I convert BF16 to FP32 in the kernel for processing. Given more time, I could have written a native BF16 kernel that would have achieved the same Kernel Bandwidth as `8_final_fp32`.

### Future Optimization Scope
1. **Cooperative Groups**: I did not explore CUDA's Cooperative Groups during this exercise. I believe the `AtomicAdd` could be optimized with Cooperative Groups.
2. **CUDA Streams**: Since this exercise does not care about H2D and D2H times, I didn't use streams. However, in real-world, CUDA streams can speedup data transfer/overlap computation and communication to provide significant performance gains.
