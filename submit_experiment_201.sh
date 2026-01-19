# for loop is inclusive of last number
for i in {1142..1146}
do
  sbatch run_gpu.sh $i false xgboost 20
done