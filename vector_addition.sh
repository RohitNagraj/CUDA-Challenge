CUDA_VISIBLE_DEVICES="7"

echo "Running fast_vector_addition.cu"

nvcc -O3 -arch=sm_87 fast_vector_addition.cu -o vector_add
./vector_add