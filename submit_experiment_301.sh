for i in {2342..2345}
do
  sbatch run_301.sh $i true mlp 1
  sbatch run_301.sh $i false mlp 1
done
