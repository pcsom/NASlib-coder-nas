#!/bin/bash
#SBATCH --job-name=codenas
#SBATCH --nodes=1
#SBATCH -c 24
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/naslib_%j.out

nvidia-smi
module load anaconda3/2023.03
conda run -n naslib39v2 --no-capture-output python -u run_nb201_comparison.py