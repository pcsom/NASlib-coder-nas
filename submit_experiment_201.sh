# for loop is inclusive of last number
for i in {1..150}
do
  sbatch run_gpu.sh $i false mlp 2
done