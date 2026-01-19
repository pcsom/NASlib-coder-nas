for i in {2042..2046}
do
  sbatch run_301.sh $i true xgboost 3
  sbatch run_301.sh $i false xgboost 3
done
