export CUDA_VISIBLE_DEVICES=7

# PROGRAM=1_naive
# PROGRAM=2_op_fusion
# PROGRAM=3_shmem
# PROGRAM=4_dynamic_blocksize
# PROGRAM=5_tree_reduction
# PROGRAM=6_warp_reduction
# PROGRAM=7_vectorize
# PROGRAM=8_final_fp32
PROGRAM=9_bf16

echo "Running $PROGRAM.cu"

nvcc -o $PROGRAM $PROGRAM.cu -arch=sm_87 -O3
echo "Run 1:"
./$PROGRAM
echo ""
echo "Run 2:"
./$PROGRAM
echo ""
echo "Run 3:"
./$PROGRAM
rm $PROGRAM