#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/naslib201_%j.out

export HF_HOME=/storage/ice-shared/vip-vvk/data/AOT/$USER/huggingface

# Accept arguments with defaults
SEED=${1:-242}
RUN_BASELINES=${2:-false}
SURROGATE=${3:-mlp}
TRIALS=${4:-1}

# echo each argument
echo "SEED: $SEED"
echo "RUN_BASELINES: $RUN_BASELINES"
echo "SURROGATE: $SURROGATE"
echo "TRIALS: $TRIALS"

# Build the Python command
PYTHON_CMD="python -u run_nb201_comparison.py --seed $SEED --surrogate $SURROGATE --trials $TRIALS"

# Add --run_baselines flag if true
if [ "$RUN_BASELINES" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --run_baselines"
fi

# sbatch run_gpu.sh 456 false xgboost

nvidia-smi
module load anaconda3/2023.03
conda run -n naslib39v2 --no-capture-output $PYTHON_CMD