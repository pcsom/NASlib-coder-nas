# latest did 842 to 847
for i in {1142..1146}
do
  sbatch run.sh $i false xgboost 20
done