# for loop is inclusive of last number
for i in {1..1}
do
  sbatch run_gpu.sh $i false xgboost 
done