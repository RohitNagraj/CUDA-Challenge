# CUDA-Challenge

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
(cudaMemcpy(DevicetoDevice)) and the bandwidth achieved for the . Measure only time on the
device, time spent copying data to/from the device is not important. Include timing in the code,
do not rely on profilers to report times.


For bonus points repeat for the bfloat16 data type (__nv_bfloat16).

## Challenges
1. Bank conflicts maybe?
2. Atomic add precision -> Solved with reduction in higher precision
3. Memory coalescing -> Already coalesced.
4. Reduction is making the code 3x slower