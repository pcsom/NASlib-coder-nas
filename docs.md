`run_nb201_comparison.py`: Patch utilities present in the codebase and compare the BANANAS surrogate-assisted NAS algorithm mainly in two different modes: using a default Path encoding to convert NAS-Bench-201 neural architectures into numerical representations, versus using COLE (our process) to convert neural architectures into numerical representations. Other configuration changes are also supported, such as the regressor model used, NAS search parameters, and a non-surrogate-assisted baseline.
  - This file can also run multiple trials at once. Each experiment is started with a different random seed. The seed used for the Nth trial is equal to N * (original seed).

`submit_experiment_201.sh`: Execute `run_nb201_comparison.py` multiple times simultaneously, with each execution having a different original random seed.

`naslib/predictors/llm_enhanced_201.py`: Provides functionality to retrieve pre-computed COLE for any NAS-Bench-201 architecture. Acts as a wrapper to other predictors (MLP, Xgboost, etc).

`naslib/predictors/llm_enhanced_101.py`: Provides functionality to compute COLE for NAS-Bench-101 architectures at runtime (dynamically). Acts as a wrapper to other predictors (MLP, Xgboost, etc).

`results_average.py`: Provides visualization and statistical testing capability for executed NAS trials. The `start_seed` and `trials` variables should match the original seed and number of trials provided to `run_nb201_comparison.py` respectively. The `num_submissions` variable should match the quantity of submissions of `run_nb201_comparison.py` you executed when running `submit_experiment_201.sh`.
