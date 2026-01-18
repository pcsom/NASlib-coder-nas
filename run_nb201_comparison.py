import os

os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
import numpy as np
import logging
from scipy.stats import norm
import sys
import time
import types
import copy
import argparse

# --- PARSE CUSTOM ARGUMENTS FIRST (before any NASLib imports) ---
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=242, help='Random seed for reproducibility')
parser.add_argument('--run_baselines', action='store_true', help='Run baseline experiments')
parser.add_argument('--surrogate', type=str, default='mlp', choices=['xgboost', 'mlp'], help='Surrogate model type')
custom_args, remaining = parser.parse_known_args()

# Update sys.argv to only contain args that NASLib's parser understands
# NASLib parser accepts --seed, so we keep it
sys.argv = [sys.argv[0], '--seed', str(custom_args.seed)] + remaining

# NOW import NASLib (which will parse the cleaned sys.argv)
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

import matplotlib.pyplot as plt

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
config.search_space = "nasbench201" 
config.out_dir = "/home/hice1/psomu3/scratch/codenas/NASLib/results_nb201" # New output dir
config.optimizer = "" 
config.search.seed = args.seed
config.seed = args.seed
config.save_arch_weights = False
config.search.num_init = 20
config.search.k = 10
config.search.epochs = 40*config.search.k + config.search.num_init
config.search.num_candidates = 200
config.out_dir = "run_nb201"
config.debug_predictor = True
config.search.num_ensemble = 3
config.search.num_arches_to_mutate = 5
config.search.max_mutations = 1
config.search.checkpoint_freq = 10000

RUN_BASELINES = args.run_baselines
RUN_ALL = True
SURROGATE = args.surrogate  # Options: 'xgboost' or 'mlp'

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
    from scipy.stats import kendalltau

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
            
            # Log the metric
            self.surrogate_test_metrics.append(tau)
            print(f"Epoch {epoch}: Surrogate Test Kendall Tau = {tau:.4f}")

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
# STRICTLY MINIMAL PATCH: SPEED + ACCURACY FIX
# =============================================================================
import torch
import torch.nn as nn
from naslib.search_spaces.core import primitives as core_ops
from naslib.search_spaces.nasbench201 import primitives as nb201_ops
import naslib.search_spaces.nasbench201.conversions as conversions_mod
from naslib.search_spaces import NasBench201SearchSpace

print("[Patch] Applying Minimal Fixes...")

# -----------------------------------------------------------------------------
# 1. SPEED FIX: Replace the Heavy Primitives
# -----------------------------------------------------------------------------
# We replace the __init__ methods of the heavy classes. 

def fast_relu_init(self, C_in, C_out, kernel_size, stride=1, affine=True, track_running_stats=True, **kwargs):
    super(core_ops.ReLUConvBN, self).__init__(locals())
    self.kernel_size = kernel_size  # Required for get_op_name
    self.stride = stride
    self.C_in = C_in
    self.C_out = C_out
    # Speedup: No nn.Conv2d created here.

def fast_avg_pool_init(self, kernel_size, stride=1, affine=True, **kwargs):
    super(core_ops.AvgPool1x1, self).__init__(locals())
    self.kernel_size = kernel_size  # Required for get_op_name
    self.stride = stride

def fast_stem_init(self, C_in, C_out, **kwargs):
    super(core_ops.Stem, self).__init__(locals())
    self.C_in = C_in
    self.C_out = C_out

def fast_resnet_init(self, C_in, C_out, stride, affine=True, track_running_stats=True, **kwargs):
    super(nb201_ops.ResNetBasicblock, self).__init__(locals())
    self.stride = stride
    self.C_in = C_in
    self.C_out = C_out

# Apply the swaps in-place.
core_ops.ReLUConvBN.__init__ = fast_relu_init
core_ops.AvgPool1x1.__init__ = fast_avg_pool_init
core_ops.Stem.__init__ = fast_stem_init
nb201_ops.ResNetBasicblock.__init__ = fast_resnet_init

print("[Patch] Primitives are now lightweight. Graph creation should be <0.001s.")

# =============================================================================
# OPTIMIZED PATCH: HOLLOW SEARCH SPACE (SAFE MODE)
# =============================================================================
def fast_hollow_init(self, n_classes=10, in_channels=3):
    """
    Replacement __init__ that skips building the graph topology.
    """
    # Initialize parent without heavy setup
    super(NasBench201SearchSpace, self).__init__()
    
    # Essential attributes for NASLib
    self.num_classes = n_classes
    self.op_indices = None
    self.max_epoch = 199
    self.in_channels = in_channels
    self.space_name = "nasbench201"
    self.labeled_archs = None
    
    # CRITICAL: Disables graph interaction
    self.instantiate_model = False 
    self.sample_without_replacement = False

# Override get_op_indices to never look at the (non-existent) graph
def safe_get_op_indices(self):
    if self.op_indices is None:
        # Fallback if accessed before setting (should not happen in Bananas)
        raise ValueError("Attempted to access op_indices of uninitialized Hollow Graph")
    return self.op_indices

# Apply patches
NasBench201SearchSpace.__init__ = fast_hollow_init
NasBench201SearchSpace.get_op_indices = safe_get_op_indices

print("[Patch] NasBench201SearchSpace is now hollow and safe.")



def run_experiment(optimizer_name, predictor_cls=None, predictor_kwargs=None):
    p_name = predictor_cls.__name__ if predictor_cls else "Default"
    print(f"\n\n>>> RUNNING: {optimizer_name} (Predictor: {p_name}) <<<")
    
    if p_name == "LLM_NB201_Predictor":
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

    # Access the optimizer instance from the trainer
    optimizer = trainer.optimizer

    # Retrieve the stored metrics
    if hasattr(optimizer, 'surrogate_test_metrics') and len(optimizer.surrogate_test_metrics) > 0:
        metrics = optimizer.surrogate_test_metrics
        epochs = range(len(metrics))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics, marker='o', linestyle='-', color='b', label='Kendall Tau')
        plt.title(f'Surrogate Generalization on Test Set ({config.optimizer})')
        plt.xlabel('Search Iterations (Model Updates)')
        plt.ylabel('Kendall Tau')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(config.save, "surrogate_test_accuracy.png")
        plt.savefig(plot_path)
        print(f"Surrogate accuracy plot saved to {plot_path}")
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
    elif SURROGATE == "mlp":
        run_experiment(
            "bananas",
            predictor_cls=CustomMLP,
            predictor_kwargs={
                "encoding_type": EncodingType.PATH, # Standard graph encoding
                "ss_type": "nasbench201",
                "num_layers": 3,
                "layer_width": 128,
                "batch_size": 32,
                "lr": 0.001,
                "epochs": 200, 
                "loss": "mse"  # or 'mse'
            }
        )

# 3c. "Ours" (Bananas with LLM Embeddings + XGBoost Predictor)
if not RUN_BASELINES or RUN_ALL:
    if SURROGATE == "xgboost":
        run_experiment(
            "bananas",
            predictor_cls=LLM_NB201_Predictor,
            predictor_kwargs={
                "base_predictor_cls": CustomXGBoost,
                "corpus_path": '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded.csv',
                "embedding_col": 'codellama_python_7b_pytorch_code_embedding',
                "use_pca": True,
                "pca_components": 128,
                # Arguments for CustomXGBoost (passed via **kwargs)
                "ss_type": "nasbench201",
                "hparams_from_file": False,
                "nthread": 4,
                "device": "cuda",
                "tree_method": "hist"
            }
        )
    elif SURROGATE == "mlp":
        run_experiment(
            "bananas",
            predictor_cls=LLM_NB201_Predictor,
            predictor_kwargs={
                "base_predictor_cls": CustomMLP,
                "corpus_path": '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded.csv',
                "embedding_col": 'codellama_python_7b_pytorch_code_embedding',
                "use_pca": True,
                "pca_components": 64,
                # --- MLP Specific Hyperparams ---
                "num_layers": 3,
                "layer_width": 128,
                "batch_size": 32,
                "lr": 0.001,
                "epochs": 200, 
                "loss": "mse"  # or 'mse'
            }
        )