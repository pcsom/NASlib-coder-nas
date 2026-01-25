import os

os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
import numpy as np
import logging
from scipy.stats import norm, kendalltau
import sys
import time
import types
import copy
import argparse
import pandas as pd

# --- PARSE CUSTOM ARGUMENTS FIRST (before any NASLib imports) ---
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=242, help='Random seed for reproducibility')
parser.add_argument('--run_baselines', action='store_true', help='Run baseline experiments')
parser.add_argument('--surrogate', type=str, default='mlp', choices=['xgboost', 'mlp'], help='Surrogate model type')
parser.add_argument('--trials', type=int, default=1, help='Number of trials to run')
custom_args, remaining = parser.parse_known_args()

# Update sys.argv to only contain args that NASLib's parser understands
# NASLib parser accepts --seed, so we keep it
sys.argv = [sys.argv[0], '--seed', str(custom_args.seed)] + remaining

# NOW import NASLib (which will parse the cleaned sys.argv)
from naslib import utils 
from naslib.utils import get_dataset_api, create_exp_dir
from naslib.utils.encodings import EncodingType
from naslib.search_spaces import NasBench101SearchSpace 
from naslib.optimizers import RegularizedEvolution, Bananas, Npenas
from naslib.defaults.trainer import Trainer

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.trees.xgb import XGBoost
from naslib.predictors.gp import VarSparseGPPredictor, GPPredictor
from naslib.predictors.llm_enhanced_101 import LLM_NB101_Predictor 
from naslib.predictors.mlp import MLPPredictor

from naslib.optimizers.discrete.bananas import optimizer as bananas_opt
from naslib.optimizers.discrete.bananas import acquisition_functions as acq_funcs

import matplotlib.pyplot as plt
import json

# Store custom args for use later in the script
args = custom_args

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
        # Start timestamp
        start = time.time()
        # Re-apply custom hyperparams just in case fit() tries to reset them
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)
        
        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)

        # End timestamp
        end = time.time()
        print(f"[CustomXGBoost] Training completed in {end - start:.2f} seconds.")
        return result

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
        # Start timestamp
        start = time.time()
        # Ensure custom hyperparams persist even if fit() tries to reset them
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        
        self.hyperparams.update(self.custom_hyperparams)
        
        result = super().fit(xtrain, ytrain, train_info=train_info, epochs=self.hyperparams["epochs"], loss=self.hyperparams["loss"], **kwargs)
        # End timestamp
        end = time.time()
        print(f"[CustomMLP] Training completed in {end - start:.2f} seconds.")
        return result

# --- CONFIGURATION ---
config = utils.get_config_from_args(config_type="nas")
config.dataset = "cifar10"
config.search_space = "nasbench101" 
config.out_dir = "/home/hice1/psomu3/scratch/codenas/NASLib/results_nb101" # New output dir
config.optimizer = "" 
config.search.seed = args.seed
config.seed = args.seed
config.save_arch_weights = False
config.search.num_init = 20
config.search.k = 10
config.search.epochs = 500
config.search.num_candidates = 500
config.out_dir = "run_nb101"
config.debug_predictor = True
config.search.num_ensemble = 3
config.search.num_arches_to_mutate = 15
config.search.max_mutations = 1
config.search.checkpoint_freq = 10000

RUN_BASELINES = args.run_baselines
RUN_ALL = True
SURROGATE = args.surrogate  # Options: 'xgboost' or 'mlp'
NUM_TRIALS = args.trials

print("RUN_BASELINES:", RUN_BASELINES)
print("SURROGATE:", SURROGATE)

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

original_acq_factory = acq_funcs.acquisition_function

def batch_acquisition_function(ensemble, ytrain, acq_fn_type="its", explore_factor=0.5, ei_calibration_factor=5.0):
    """
    Batched wrapper that returns a function capable of processing 
    a list of architectures in one shot.
    """
    
    # Define helper to handle batch statistics (N, Ensemble_Size) -> (N,)
    def get_stats(preds): 
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    # 2. Define the ACTUAL execution logic (The thing that runs on GPU)
    def batched_executor(archs, info=None):
        t0 = time.time()
        
        # A. Batch Inference (The Bottleneck)
        # ensemble.query passes the full list 'archs' to the predictor
        # This triggers ONE gpu call instead of 2000
        preds = ensemble.query(archs, info) 
        
        # B. Acquisition Math
        mean, std = get_stats(preds)
        
        if acq_fn_type == "its":
            scores = np.random.normal(mean, std)
        elif acq_fn_type == "ucb":
            scores = mean + explore_factor * std
        elif acq_fn_type == "ei":
            fs = std / ei_calibration_factor
            gam = (mean - ytrain.max()) / fs
            scores = fs * (gam * norm.cdf(gam) + norm.pdf(gam))
        else:
            # Fallback for unknown types (rare)
            scores = original_acq_factory(ensemble, ytrain, acq_fn_type)(archs, info)

        t1 = time.time()
        print(f"[Batch Acquisition] Processed {len(archs)} candidates in {t1-t0:.4f}s")
        return scores

    # 3. Return the batched callable
    return batched_executor

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

# --- MONKEY PATCH TO DEBUG TIMING ---
def debug_new_epoch(self, epoch):
    import time
    
    if epoch < self.num_init:
        model = self._sample_new_model()
        self._set_scores(model)
    else:
        if len(self.next_batch) == 0:
            # 1. Train
            print("[Debug] Starting Fit...")
            t0 = time.time()
            xtrain, ytrain = self._get_train()
            self.ensemble = self._get_ensemble()
            
            # Setup semi-supervised if needed (omitted for brevity as self.semi usually False)
            
            self.ensemble.fit(xtrain, ytrain)
            t1 = time.time()
            print(f"[Debug] Fit finished in {t1-t0:.2f}s")

            # 2. Acquisition Setup
            acq_fn = bananas_opt.acquisition_function(
                ensemble=self.ensemble, ytrain=ytrain, acq_fn_type=self.acq_fn_type
            )

            # 3. Candidate Generation
            print(f"[Debug] Generating {self.num_candidates} candidates...")
            t2 = time.time()
            candidates = self._get_new_candidates(ytrain=ytrain)
            t3 = time.time()
            print(f"[Debug] Generated candidates in {t3-t2:.2f}s")

            # 4. Selection (Acquisition)
            print("[Debug] Selecting best candidates...")
            self.next_batch = self._get_best_candidates(candidates, acq_fn)
            t4 = time.time()
            print(f"[Debug] Selection finished in {t4-t3:.2f}s")
            
            # Compute Kendall Tau on test set (only when ensemble is retrained)
            pred_scores = self.ensemble.query(self.test_data)  # Shape: (num_ensemble, num_test)
            # Aggregate ensemble predictions by taking mean across ensemble members
            mean_pred_scores = np.mean(pred_scores, axis=0)  # Shape: (num_test,)
            
            # Calculate Kendall Tau (Ranking Correlation)
            tau, _ = kendalltau(self.test_accuracies, mean_pred_scores)
            
            # Calculate MSE (Mean Squared Error)
            mse = np.mean((np.array(self.test_accuracies) - mean_pred_scores) ** 2)
            
            # Log the metrics
            self.surrogate_test_metrics.append(tau)
            self.surrogate_test_metrics_mse.append(mse)
            print(f"Epoch {epoch}: Surrogate Test Kendall Tau = {tau:.4f}, MSE = {mse:.6f}")

        # 5. Evaluation
        print(f"[Debug] Evaluating architecture {len(self.next_batch)}...")
        t5 = time.time()
        model = self.next_batch.pop()
        self._set_scores(model)
        print(f"[Debug] Evaluation finished in {time.time()-t5:.2f}s")

# Apply the patch
bananas_opt.Bananas.new_epoch = debug_new_epoch
print("Monkey-patched Bananas.new_epoch for profiling.")




# =============================================================================
# OPTIMIZED PATCH: HOLLOW SEARCH SPACE FOR NB101 (DICT-SPEC)
# =============================================================================
from naslib.search_spaces.nasbench101.graph import NasBench101SearchSpace
from naslib.search_spaces.nasbench101.conversions import convert_spec_to_tuple
import numpy as np
import copy

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
NUM_VERTICES = 7

def fast_hollow_init_101(self, n_classes=10):
    """
    Hollow Init for NB101. 
    Skips graph construction and model instantiation.
    """
    # 1. Init Parent (lightweight)
    super(NasBench101SearchSpace, self).__init__()
    
    # 2. Set Attributes
    self.num_classes = n_classes
    self.space_name = "nasbench101"
    self.spec = None
    self.labeled_archs = None
    
    # CRITICAL: Disable model instantiation to prevent expensive conversion
    self.instantiate_model = False 
    self.sample_without_replacement = False
    
    # 3. Skip Graph Edge creation (self.add_edge(1, 2))

def hollow_sample_random_architecture_101(self, dataset_api, load_labeled=False):
    """
    Samples a random Spec (matrix, ops) directly as a Dictionary.
    """
    if load_labeled:
        # Re-use existing logic but ensure it sets spec without model
        self.sample_random_labeled_architecture()
        return

    while True:
        matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT

        spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
        if dataset_api["nb101_data"].is_valid(spec):
            break

    # Set the spec directly as a DICT
    self.spec = {"matrix": matrix, "ops": ops}

def hollow_mutate_101(self, parent, dataset_api, edits=1):
    """
    Mutates the Spec directly.
    """
    parent_spec = parent.get_spec()
    
    # Handle parent spec being dict or object
    if isinstance(parent_spec, dict):
        matrix, ops = parent_spec['matrix'], parent_spec['ops']
    else:
        matrix, ops = parent_spec.matrix, parent_spec.ops

    for _ in range(edits):
        while True:
            new_matrix = copy.deepcopy(matrix)
            new_ops = copy.deepcopy(ops)
            
            # Mutate Edges
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if np.random.random() < 1 / NUM_VERTICES:
                        new_matrix[src][dst] = 1 - new_matrix[src][dst]
            
            # Mutate Ops
            for ind in range(1, NUM_VERTICES - 1):
                if np.random.random() < 1 / len(OPS):
                    available = [op for op in OPS if op != new_ops[ind]]
                    new_ops[ind] = np.random.choice(available)
            
            # Verify validity
            new_spec_obj = dataset_api['api'].ModelSpec(new_matrix, new_ops)
            if dataset_api['nb101_data'].is_valid(new_spec_obj):
                break
    
    self.spec = {'matrix': new_matrix, 'ops': new_ops}

def hollow_get_hash_101(self):
    # convert_spec_to_tuple handles dicts correctly via key access
    return convert_spec_to_tuple(self.spec)

def hollow_clone_101(self):
    new_arch = NasBench101SearchSpace()
    new_arch.spec = copy.deepcopy(self.spec)
    return new_arch

# --- APPLY PATCHES ---
NasBench101SearchSpace.__init__ = fast_hollow_init_101
NasBench101SearchSpace.sample_random_architecture = hollow_sample_random_architecture_101
NasBench101SearchSpace.mutate = hollow_mutate_101
NasBench101SearchSpace.get_hash = hollow_get_hash_101
NasBench101SearchSpace.clone = hollow_clone_101

print("[Patch] NAS-Bench-101 is now hollow and Spec-driven.")





for i in range(NUM_TRIALS):
    print(f"\n\n=== TRIAL {i+1}/{NUM_TRIALS} ===")
    config.search.seed = args.seed*(i+1)
    config.seed = config.search.seed

    def run_experiment(optimizer_name, predictor_cls=None, predictor_kwargs=None):
        p_name = predictor_cls.__name__ if predictor_cls else "Default"
        print(f"\n\n>>> RUNNING: {optimizer_name} (Predictor: {p_name}) <<<")

        if p_name == "LLM_NB101_Predictor":
            base_pred = predictor_kwargs['base_predictor_cls'].__name__
            config.optimizer = f"{p_name}_{base_pred}_{optimizer_name}"
        else:
            config.optimizer = f"{p_name}_{optimizer_name}"
        set_config_save()

        # 1. Select Optimizer
        if optimizer_name == "rea":
            optimizer = RegularizedEvolution(config)
        elif optimizer_name == "bananas":
            optimizer = Bananas(config)
        elif optimizer_name == "npenas":
            optimizer = Npenas(config)
        
        # 2. Setup Search Space (NB101)
        search_space = NasBench101SearchSpace()
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
        
        # Access the optimizer instance from the trainer
        optimizer = trainer.optimizer

        # Retrieve the stored metrics
        if hasattr(optimizer, 'surrogate_test_metrics') and len(optimizer.surrogate_test_metrics) > 0:
            metrics_tau = optimizer.surrogate_test_metrics
            metrics_mse = optimizer.surrogate_test_metrics_mse if hasattr(optimizer, 'surrogate_test_metrics_mse') else []
            epochs = list(range(len(metrics_tau)))

            # Save metrics to JSON file for later analysis
            # Create a unique experiment identifier
            predictor_type = p_name if p_name != "LLM_NB101_Predictor" else f"LLM_{predictor_kwargs['base_predictor_cls'].__name__}"
            experiment_id = f"{predictor_type}_{optimizer_name}"
            
            metrics_data = {
                'experiment_id': experiment_id,
                'optimizer_type': optimizer_name,
                'predictor_class': p_name,
                'base_predictor': predictor_kwargs.get('base_predictor_cls').__name__ if p_name == "LLM_NB101_Predictor" else None,
                'surrogate_type': SURROGATE,
                'uses_llm': p_name == "LLM_NB101_Predictor",
                'dataset': config.dataset,
                'seed': config.search.seed,
                'trial': i,
                'epochs': epochs,
                'kendall_tau': metrics_tau,
                'mse': metrics_mse,
                'config_str': config.optimizer
            }
            
            metrics_path = os.path.join(config.save, f"surrogate_metrics_trial_{i}_seed_{config.search.seed}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"Surrogate metrics saved to {metrics_path}")

            # Plot individual trial with dual panels
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot Kendall Tau
            ax1.plot(epochs, metrics_tau, marker='o', linestyle='-', color='b', label='Kendall Tau')
            ax1.set_title(f'Surrogate Kendall Tau on Test Set ({config.optimizer})')
            ax1.set_xlabel('Search Iterations (Model Updates)')
            ax1.set_ylabel('Kendall Tau')
            ax1.grid(True)
            ax1.legend()
            
            # Plot MSE (if available)
            if metrics_mse:
                ax2.plot(epochs, metrics_mse, marker='s', linestyle='-', color='r', label='MSE')
                ax2.set_title(f'Surrogate MSE on Test Set ({config.optimizer})')
                ax2.set_xlabel('Search Iterations (Model Updates)')
                ax2.set_ylabel('MSE')
                ax2.grid(True)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'MSE data not available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(config.save, f"surrogate_test_metrics_trial_{i}_seed_{config.search.seed}.png")
            plt.savefig(plot_path)
            print(f"Surrogate metrics plot saved to {plot_path}")
            plt.close()
    
            return trainer.optimizer.history





    # --- EXPERIMENTS ---

    # 1. The "True" Baseline (No Predictor)
    if RUN_BASELINES or RUN_ALL:
        run_experiment("rea")

    # 2c. The "Competitor" (Bananas with XGBoost Predictor)
    if RUN_BASELINES or RUN_ALL:
        if SURROGATE == "xgboost":
            run_experiment(
                "bananas",
                predictor_cls=CustomXGBoost,
                predictor_kwargs={
                    "encoding_type": EncodingType.PATH, # Standard graph encoding
                    "ss_type": "nasbench101",
                    "hparams_from_file": False,
                    "nthread": 4,
                    "device": "cuda",
                    "tree_method": "hist"
                    # Hyperparams from NASLib Paper Table 2
                    # "max_depth": 6,
                    # "learning_rate": 0.3,
                }
            )
        elif SURROGATE == "mlp":
            run_experiment(
                "bananas",
                predictor_cls=CustomMLP,
                predictor_kwargs={
                    "encoding_type": EncodingType.PATH, # Standard graph encoding
                    "ss_type": "nasbench101",
                    "num_layers": 3,
                    "layer_width": 128,
                    "batch_size": 32,
                    "lr": 0.001,
                    "epochs": 200, 
                    "loss": "mse"  # or 'mse'
                }
            )


    if not RUN_BASELINES or RUN_ALL:
        if SURROGATE == "xgboost":
            run_experiment(
                "bananas",
                predictor_cls=LLM_NB101_Predictor,
                predictor_kwargs={
                    "base_predictor_cls": CustomXGBoost,
                    "ss_type": "nasbench101",
                    "hparams_from_file": False,
                    "nthread": 4,
                    "device": "cuda",
                    "tree_method": "hist",
                    "use_pca": False,
                    "pca_components": 128
                }
            )
        elif SURROGATE == "mlp":
            run_experiment(
                "bananas",
                predictor_cls=LLM_NB101_Predictor,
                predictor_kwargs={
                    "base_predictor_cls": CustomMLP,
                    "ss_type": "nasbench101",
                    "use_pca": True,
                    "pca_components": 128,
                    "num_layers": 3,
                    "layer_width": 128,
                    "batch_size": 32,
                    "lr": 0.001,
                    "epochs": 200, 
                    "loss": "mse"  # or 'mse'
                }
            )