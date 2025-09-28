PROGRAM=2_naive

nvcc -o $PROGRAM $PROGRAM.cu
echo "Run 1:"
./$PROGRAM
echo "Run 2:"
./$PROGRAM
echo "Run 3:"
./$PROGRAM