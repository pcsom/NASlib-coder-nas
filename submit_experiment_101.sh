for i in {1247..1251}
do
  sbatch run_101.sh $i false xgboost 10
done