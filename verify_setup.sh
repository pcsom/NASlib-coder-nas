#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

failed=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}[+]${NC} $1"
    else
        echo -e "  ${RED}[-]${NC} $1"
        failed=1
    fi
}

check_import() {
    python -c "import $1" 2>/dev/null
    check "$2"
}

check_dir() {
    [ -d "$1" ]
    check "$2 ... $1"
}

check_file() {
    [ -f "$1" ]
    check "$2 ... $1"
}

echo
echo "checking naslib setup..."
echo

echo "core stuff:"
python -c "import sys; exit(0 if sys.version_info[:2] == (3,9) else 1)" 2>/dev/null
check "python 3.9"
check_import torch "torch"
check_import torchvision "torchvision"
check_import torchaudio "torchaudio"
check_import torch "cuda support (will work on compute nodes)"
echo

echo "main dependencies:"
check_import numpy "numpy"
check_import scipy "scipy"
check_import ConfigSpace "ConfigSpace"
check_import transformers "transformers"
check_import accelerate "accelerate"
check_import yaml "pyyaml"
check_import networkx "networkx"
check_import sklearn "scikit-learn"
check_import skimage "scikit-image"
check_import pandas "pandas"
check_import tornado "tornado"
check_import seaborn "seaborn"
echo

echo "nasbench stuff:"
check_import nasbench_pytorch "nasbench_pytorch"
check_import nasbench301 "nasbench301"
check_import fvcore "fvcore"
echo

echo "ml packages:"
check_import lightgbm "lightgbm"
check_import xgboost "xgboost"
check_import ngboost "ngboost"
check_import emcee "emcee"
check_import pybnn "pybnn"
check_import grakel "grakel"
echo

echo "other tools:"
check_import pytest "pytest"
check_import tqdm "tqdm"
check_import transforms3d "transforms3d"
check_import gdown "gdown"
echo

echo "note: skipping bitsandbytes, pyro-ppl, pytorch-msssim, tensorwatch"
echo "      (these can hang on login nodes but work on compute nodes)"
echo

echo "directories:"
check_dir "$HOME/slurm_logs" "slurm_logs"
check_dir "naslib/data" "naslib/data"
echo

echo "benchmark data (optional):"
check_file "naslib/data/nb201_all.pickle" "nb201"
check_dir "naslib/data/nb_models_1.0" "nb301 models"
echo

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}Looks good!${NC} Everything required is installed."
    echo
    echo "Notes:"
    echo "  - cuda will work when you submit jobs to compute nodes"
    echo "  - nb301 models are optional, download if you need them"
    echo "  - you can test: python test_benchmark_apis.py --all"
    exit 0
else
    echo -e "${RED}Hmm, something's missing.${NC} Check the failures above."
    echo
    exit 1
fi
