CUDA_VISIBLE_DEVICES="7"

# PROGRAM=1_naive
PROGRAM=2_op_fusion
# PROGRAM=3_op_fusion
# PROGRAM=4_shmem
# PROGRAM=5_vectorize

echo "Running $PROGRAM.cu"

nvcc -o $PROGRAM $PROGRAM.cu -arch=sm_87
echo "Run 1:"
./$PROGRAM
echo "Run 2:"
./$PROGRAM
# echo "Run 3:"
# ./$PROGRAM
rm $PROGRAM