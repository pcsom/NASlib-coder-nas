import os

os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
import numpy as np
import logging
from scipy.stats import norm
import sys
import types
import copy
import argparse

from naslib import utils 
from naslib.utils import get_dataset_api, create_exp_dir
from naslib.utils.encodings import EncodingType
from naslib.search_spaces import NasBench201SearchSpace 
from naslib.optimizers import RegularizedEvolution, Bananas, Npenas
from naslib.defaults.trainer import Trainer

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.trees.xgb import XGBoost
from naslib.predictors.gp import VarSparseGPPredictor, GPPredictor
from naslib.predictors.llm_enhanced_201 import LLM_NB201_Predictor 
from naslib.predictors.mlp import MLPPredictor


from naslib.optimizers.discrete.bananas import optimizer as bananas_opt
from naslib.optimizers.discrete.bananas import acquisition_functions as acq_funcs

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser(description='Run NASBench201 comparison experiments')
parser.add_argument('--seed', type=int, default=242, help='Random seed for reproducibility')
args = parser.parse_args()

class CustomXGBoost(XGBoost):
    def __init__(self, **kwargs):
        # 1. Define arguments allowed by BaseTree.__init__
        base_valid_args = ['encoding_type', 'ss_type', 'zc', 'zc_only', 
                           'hpo_wrapper', 'hparams_from_file']
        
        # 2. Split kwargs: 
        # - base_args go to super().__init__
        # - hyperparams go into the model config
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        # 3. Initialize Parent
        super().__init__(**base_args)

        # 4. Apply Hyperparameters
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        
        # Inject our custom settings (nthread, max_depth, learning_rate, etc.)
        self.hyperparams.update(self.custom_hyperparams)
        
        print(f"[CustomXGBoost] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        # Re-apply custom hyperparams just in case fit() tries to reset them
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)
        
        return super().fit(xtrain, ytrain, train_info, params, **kwargs)
    

class CustomMLP(MLPPredictor):
    def __init__(self, **kwargs):
        # 1. Define arguments allowed by BasePredictor.__init__
        # We must filter these out so we don't pass 'lr' or 'epochs' to the parent class
        base_valid_args = ['encoding_type', 'ss_type', 'zc', 'zc_only', 
                           'hpo_wrapper', 'hparams_from_file', 'config']
        
        # 2. Split kwargs into Base args and Hyperparameters
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        # 3. Initialize Parent (MLPPredictor)
        super().__init__(**base_args)

        # 4. Inject Hyperparameters
        if self.hyperparams is None:
            # Load defaults if not already present
            self.hyperparams = self.default_hyperparams.copy()
        
        # Update with your custom values (e.g., batch_size, lr)
        self.hyperparams.update(self.custom_hyperparams)
        
        print(f"[CustomMLP] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        # Ensure custom hyperparams persist even if fit() tries to reset them
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        
        self.hyperparams.update(self.custom_hyperparams)
        
        return super().fit(xtrain, ytrain, train_info, params, **kwargs)

# --- CONFIGURATION ---
config = utils.get_config_from_args(config_type="nas")
config.dataset = "cifar100"
config.search_space = "nasbench201" 
config.out_dir = "/home/hice1/psomu3/scratch/codenas/NASLib/results_nb201" # New output dir
config.optimizer = "" 
config.search.seed = args.seed
config.save_arch_weights = False
config.search.num_init = 20
config.search.k = 5
config.search.epochs = 25*config.search.k + config.search.num_init
config.search.num_candidates = 1000
config.out_dir = "run_nb201"
config.debug_predictor = True
config.search.num_ensemble = 3
config.search.num_arches_to_mutate = 45
config.search.max_mutations = 1
config.search.checkpoint_freq = 10000

RUN_BASELINES = True
RUN_ALL = True

if RUN_ALL:
    print("Running Both Baselines and LLM method.")
elif RUN_BASELINES:
    print("Running Baselines only.")
else:
    print("Running LLM method only.")

# config printing
print("Experiment Configuration:")
print(f"Dataset: {config.dataset}")
print(f"Search Space: {config.search_space}")
print(f"Seed: {config.search.seed}")
print(f"Number of Initial Samples: {config.search.num_init}")
print(f"Expansion Size (k): {config.search.k}")
print(f"Number of Search Epochs: {config.search.epochs}")
print(f"Number of Candidates: {config.search.num_candidates}")
print(f"Size of Ensemble: {config.search.num_ensemble}")

def set_config_save():
    global config
    config.save = os.path.join(config.out_dir, config.search_space, config.dataset, config.optimizer, str(config.search.seed))

def write_config_to_file():
    global config
    config_save_path = os.path.join(config.save, "config.txt")
    with open(config_save_path, "w") as f:
        f.write(str(config))
    print(f"Configuration saved to {config_save_path}")

def batch_acquisition_function(ensemble, ytrain, acq_fn_type="its", explore_factor=0.5, ei_calibration_factor=5.0):
    # Minimal wrapper to handle batch inputs (axis=0)
    def get_stats(preds): return np.mean(preds, axis=0), np.std(preds, axis=0)

    if acq_fn_type == "its":
        return lambda archs, info=None: np.random.normal(*get_stats(ensemble.query(archs, info)))
    elif acq_fn_type == "ucb":
        return lambda archs, info=None: (lambda m, s: m + explore_factor * s)(*get_stats(ensemble.query(archs, info)))
    elif acq_fn_type == "ei":
        def ei(archs, info=None):
            m, s = get_stats(ensemble.query(archs, info))
            fs = s / ei_calibration_factor
            gam = (m - ytrain.max()) / fs
            return fs * (gam * norm.cdf(gam) + norm.pdf(gam))
        return ei
    return acq_funcs.acquisition_function(ensemble, ytrain, acq_fn_type, explore_factor, ei_calibration_factor)

def _get_best_candidates_batch(self, candidates, acq_fn):
    # Pass full list to acq_fn instead of loop
    info = [{'zero_cost_scores': c.zc_scores} for c in candidates] if self.zc and len(self.train_data) <= self.max_zerocost else None
    values = acq_fn([c.arch for c in candidates], info)
    return [candidates[i] for i in np.argsort(values)[-self.k:]]

acq_funcs.acquisition_function = batch_acquisition_function
bananas_opt.acquisition_function = batch_acquisition_function
bananas_opt.Bananas._get_best_candidates = _get_best_candidates_batch
print("Patched Bananas optimizer for batch acquisition function.")

# --- LOAD API ---
dataset_api = get_dataset_api(config.search_space, config.dataset)

def run_experiment(optimizer_name, predictor_cls=None, predictor_kwargs=None):
    p_name = predictor_cls.__name__ if predictor_cls else "Default"
    print(f"\n\n>>> RUNNING: {optimizer_name} (Predictor: {p_name}) <<<")
    
    config.optimizer = f"{p_name}_{optimizer_name}"
    set_config_save()

    # 1. Select Optimizer
    if optimizer_name == "rea":
        optimizer = RegularizedEvolution(config)
    elif optimizer_name == "bananas":
        optimizer = Bananas(config)
    elif optimizer_name == "npenas":
        optimizer = Npenas(config)
    
    # 2. Setup Search Space (NB201)
    search_space = NasBench201SearchSpace()
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    
    # 3. INJECT CUSTOM PREDICTOR
    if predictor_cls is not None:
        print(f"Injecting Custom Predictor: {p_name}")
        
        if optimizer_name == "bananas":
            # --- MONKEY PATCHING ENSEMBLE ---
            def _get_custom_ensemble(self):
                ensemble = Ensemble(num_ensemble=self.num_ensemble, ss_type=self.ss_type, predictor_type=self.predictor_type, config=self.config, zc=self.zc)
                # CHANGE: Create 3 instances matching the given custom predictor
                ensemble.ensemble = [
                    predictor_cls(**predictor_kwargs) 
                    for _ in range(self.num_ensemble)
                ]
                # try:
                #     print("Ensemble Predictor Hyperparameters:")
                #     print(ensemble.ensemble[0].default_hyperparams)
                # except:
                #     print("Predictor has no default_hyperparams attribute.")
                return ensemble

            optimizer._get_ensemble = types.MethodType(_get_custom_ensemble, optimizer)
            # --- END MONKEY PATCHING ---
        else:
            optimizer.predictor = predictor_cls(**predictor_kwargs)
    
    # 4. Run Search
    create_exp_dir(config.save)
    create_exp_dir(config.save + "/search")
    create_exp_dir(config.save + "/eval")
    write_config_to_file()
    trainer = Trainer(optimizer, config, lightweight_output=True)
    trainer.search() 
    
    return trainer.optimizer.history





# --- EXPERIMENTS ---

# 1. The "True" Baseline (No Predictor)
if RUN_BASELINES or RUN_ALL:
    run_experiment("rea")

# 2c. The "Competitor" (Bananas with XGBoost Predictor)
if RUN_BASELINES or RUN_ALL:
    run_experiment(
        "bananas",
        predictor_cls=CustomXGBoost,
        predictor_kwargs={
            "encoding_type": EncodingType.PATH, # Standard graph encoding
            "ss_type": "nasbench201",
            "hparams_from_file": False,
            "nthread": 4,
            "device": "cuda",
            "tree_method": "hist"
            # Hyperparams from NASLib Paper Table 2
            # "max_depth": 6,
            # "learning_rate": 0.3,
        }
    )

# 3c. "Ours" (Bananas with LLM Embeddings + XGBoost Predictor)
if not RUN_BASELINES or RUN_ALL:
    run_experiment(
        "bananas",
        predictor_cls=LLM_NB201_Predictor,
        predictor_kwargs={
            "base_predictor_cls": CustomXGBoost,
            "corpus_path": '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded.csv',
            "embedding_col": 'codellama_python_7b_pytorch_code_embedding',
            "use_pca": True,
            "pca_components": 64,
            # Arguments for CustomXGBoost (passed via **kwargs)
            "ss_type": "nasbench201",
            "hparams_from_file": False,
            "nthread": 4,
            "device": "cuda",
            "tree_method": "hist"
        }
    )