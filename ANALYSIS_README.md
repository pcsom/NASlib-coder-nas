# Surrogate Model Analysis Guide

## Overview

This analysis pipeline evaluates surrogate model performance across multiple trials using Kendall Tau correlation on a fixed test set. It includes statistical significance testing and publication-quality visualizations.

## Workflow

### 1. Run Experiments with Multiple Trials

Run your experiments with the `--trials` flag to execute multiple random seeds:

```bash
# Run 5 trials with XGBoost surrogate
python run_nb201_comparison.py --seed 242 --trials 5 --surrogate xgboost

# Run 5 trials with MLP surrogate  
python run_nb201_comparison.py --seed 242 --trials 5 --surrogate mlp
```

**What happens:**
- First trial generates a test set (200 architectures) and caches it
- Subsequent trials reuse the cached test set for consistent evaluation
- Each trial saves metrics to `surrogate_metrics_trial_{i}_seed_{seed}.json`

### 2. Analyze Results

After running all experiments, analyze the results:

```bash
# Basic analysis with default baseline
python analyze_surrogate_results.py \
    --results_dir run_nb201 \
    --dataset cifar10 \
    --baseline CustomXGBoost_bananas \
    --output cifar10_surrogate_comparison.png
```

**Arguments:**
- `--results_dir`: Directory containing experiment results (default: `run_nb201`)
- `--dataset`: Dataset to analyze (`cifar10`, `cifar100`, `ImageNet16-120`)
- `--baseline`: Baseline experiment name for significance testing
- `--output`: Output plot filename

### 3. Understand the Output

The analysis script generates:

#### Console Output:
```
SUMMARY STATISTICS
==================
CustomXGBoost_bananas:
  Number of trials: 5
  Final Kendall Tau: 0.6234 ± 0.0123
  95% CI: [0.5988, 0.6480]
  Min/Max across trials: [0.6012, 0.6456]

LLM_NB201_Predictor_CustomXGBoost_bananas:
  Number of trials: 5
  Final Kendall Tau: 0.7145 ± 0.0098
  95% CI: [0.6949, 0.7341]
  Min/Max across trials: [0.7023, 0.7267]

SIGNIFICANCE TESTS (Final Epoch)
=================================
LLM_NB201_Predictor_CustomXGBoost_bananas_vs_CustomXGBoost_bananas:
  Mean difference: +0.0911
  t-statistic: 5.6234
  p-value: 0.0023 **
  Significant (α=0.05): True
```

#### Plots:
1. **Left panel**: Learning curves showing Kendall Tau over search iterations
   - Lines show mean across trials
   - Shaded regions show 95% confidence intervals
   
2. **Right panel**: Final performance comparison
   - Bars show final Kendall Tau with error bars (SEM)
   - Stars indicate statistical significance vs. baseline:
     - `*`: p < 0.05
     - `**`: p < 0.01
     - `***`: p < 0.001

## Experiment Naming Convention

The optimizer name in the metrics follows this pattern:
- `Default_rea`: Regularized Evolution (no surrogate)
- `CustomXGBoost_bananas`: BANANAS with XGBoost
- `CustomMLP_bananas`: BANANAS with MLP
- `LLM_NB201_Predictor_CustomXGBoost_bananas`: BANANAS with LLM+XGBoost
- `LLM_NB201_Predictor_CustomMLP_bananas`: BANANAS with LLM+MLP

## Statistical Details

- **Confidence Intervals**: 95% CI using t-distribution (appropriate for small sample sizes)
- **Significance Testing**: Welch's t-test (doesn't assume equal variance)
- **Multiple Comparisons**: Tests baseline vs. all other methods

## Tips

1. **Run more trials**: For robust statistics, aim for at least 3-5 trials per experiment
2. **Different datasets**: Analyze each dataset separately for dataset-specific insights
3. **Custom comparisons**: Modify `perform_significance_tests()` for specific pairwise comparisons

## Example Full Workflow

```bash
# Run all experiments (this will take time!)
for seed in 42 43 44 45 46; do
    python run_nb201_comparison.py --seed $seed --surrogate xgboost
done

# Analyze CIFAR-10 results
python analyze_surrogate_results.py \
    --dataset cifar10 \
    --baseline CustomXGBoost_bananas \
    --output figures/cifar10_comparison.png

# Analyze CIFAR-100 results  
python analyze_surrogate_results.py \
    --dataset cifar100 \
    --baseline CustomXGBoost_bananas \
    --output figures/cifar100_comparison.png
```
