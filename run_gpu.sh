#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/naslib301_%j.out

# Accept seed as first argument, default to 242 if not provided
SEED=${1:-242}

nvidia-smi
module load anaconda3/2023.03
conda run -n naslib39v2 --no-capture-output python -u run_nb301_comparison.py --seed $SEED