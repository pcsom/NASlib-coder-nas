# for loop is inclusive of last number
for i in {1942..1946}
do
  sbatch run_gpu.sh $i false xgboost 2
done