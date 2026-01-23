"""
Script to analyze surrogate model performance across multiple trials.
Loads Kendall Tau metrics from JSON files, computes statistics, performs 
significance testing, and creates publication-quality plots.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import argparse

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def create_display_name(exp_id, metadata=None):
    """
    Create a user-friendly display name for experiments.
    
    Args:
        exp_id: experiment identifier
        metadata: optional metadata dict with predictor info
    
    Returns:
        str: formatted display name
    """
    if metadata is None:
        return exp_id
    
    predictor = metadata.get('predictor_class', '')
    uses_llm = metadata.get('uses_llm', False)
    base_pred = metadata.get('base_predictor', '')
    surrogate = metadata.get('surrogate_type', '')
    
    if uses_llm and base_pred:
        return f"LLM + {base_pred} ({surrogate})"
    elif predictor.startswith('Custom'):
        pred_name = predictor.replace('Custom', '')
        return f"{pred_name} ({surrogate})"
    elif predictor == 'Default':
        return "Random Evolution (No Surrogate)"
    else:
        return exp_id

def load_metrics(results_dir, dataset='cifar10'):
    """
    Load all surrogate metrics from JSON files.
    
    Returns:
        dict: {experiment_id: {'epochs': [...], 'trials': [...], 'metadata': {...}}}
    """
    metrics = defaultdict(lambda: {'epochs': None, 'trials': [], 'metadata': {}})
    
    # Find all JSON metric files
    pattern = os.path.join(results_dir, 'nasbench201', dataset, '**', 'surrogate_metrics_*.json')
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} metric files")
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Use experiment_id if available, fallback to old 'optimizer' field
            experiment_id = data.get('experiment_id', data.get('optimizer', 'unknown'))
            epochs = data['epochs']
            kendall_tau = data['kendall_tau']
            
            # Store metadata from first trial
            if not metrics[experiment_id]['metadata']:
                metrics[experiment_id]['metadata'] = {
                    'predictor_class': data.get('predictor_class', 'unknown'),
                    'optimizer_type': data.get('optimizer_type', 'unknown'),
                    'surrogate_type': data.get('surrogate_type', 'unknown'),
                    'uses_llm': data.get('uses_llm', False),
                    'base_predictor': data.get('base_predictor', None),
                    'dataset': data.get('dataset', dataset)
                }
            
            # Store epochs (should be the same for all trials)
            if metrics[experiment_id]['epochs'] is None:
                metrics[experiment_id]['epochs'] = epochs
            
            metrics[experiment_id]['trials'].append(kendall_tau)
            
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return dict(metrics)

def compute_statistics(metrics_data):
    """
    Compute mean, std, and confidence intervals for each experiment.
    
    Returns:
        dict: {experiment_name: {'epochs': [...], 'mean': [...], 'std': [...], 
                                  'ci_lower': [...], 'ci_upper': [...]}}
    """
    stats_data = {}
    
    for exp_name, data in metrics_data.items():
        epochs = data['epochs']
        trials = np.array(data['trials'])  # Shape: (n_trials, n_epochs)
        
        if len(trials) == 0:
            continue
        
        mean = np.mean(trials, axis=0)
        std = np.std(trials, axis=0, ddof=1) if len(trials) > 1 else np.zeros_like(mean)
        sem = std / np.sqrt(len(trials))  # Standard error of the mean
        
        # 95% confidence interval using t-distribution
        confidence = 0.95
        if len(trials) > 1:
            t_value = stats.t.ppf((1 + confidence) / 2, len(trials) - 1)
            ci_margin = t_value * sem
        else:
            ci_margin = np.zeros_like(mean)

        stats_data[exp_name] = {
            'epochs': epochs,
            'mean': mean,
            'std': std,
            'sem': sem,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'n_trials': len(trials)
        }
    
    return stats_data

def perform_significance_tests_per_epoch(metrics_data, baseline_name=None):
    """
    Perform pairwise significance tests between experiments at each epoch.
    
    Args:
        metrics_data: dict of experiment data
        baseline_name: name of baseline experiment to compare against (optional)
    
    Returns:
        tuple: (epoch_wise_significance, final_values)
            - epoch_wise_significance: dict[comparison][epoch] = {'p_value', 'significant', 'mean_diff', ...}
            - final_values: dict[exp_name] = array of final epoch values across trials
    """
    epoch_wise_significance = {}
    final_values = {}
    
    # Get all experiment names and determine max epochs
    experiment_names = list(metrics_data.keys())
    if not experiment_names:
        return {}, {}
    
    # Get the number of epochs (assume all experiments have same length)
    n_epochs = len(metrics_data[experiment_names[0]]['epochs'])
    
    # Extract data for all experiments
    trials_data = {}
    for exp_name, data in metrics_data.items():
        trials = np.array(data['trials'])  # Shape: (n_trials, n_epochs)
        if len(trials) > 0:
            trials_data[exp_name] = trials
            final_values[exp_name] = trials[:, -1]
    
    if baseline_name and baseline_name in trials_data:
        # Compare all experiments against baseline at each epoch
        baseline_trials = trials_data[baseline_name]
        
        for exp_name in experiment_names:
            if exp_name == baseline_name:
                continue
            
            if exp_name not in trials_data:
                continue
                
            exp_trials = trials_data[exp_name]
            comparison_key = f"{exp_name}_vs_{baseline_name}"
            epoch_wise_significance[comparison_key] = {}
            
            # Test at each epoch
            for epoch_idx in range(n_epochs):
                baseline_vals = baseline_trials[:, epoch_idx]
                exp_vals = exp_trials[:, epoch_idx]
                
                # Welch's t-test (doesn't assume equal variance)
                t_stat, p_val = stats.ttest_ind(exp_vals, baseline_vals, equal_var=False)
                
                epoch_wise_significance[comparison_key][epoch_idx] = {
                    'p_value': p_val,
                    't_statistic': t_stat,
                    'significant': p_val < 0.05,
                    'mean_diff': np.mean(exp_vals) - np.mean(baseline_vals)
                }
    else:
        # All pairwise comparisons at each epoch
        for i, exp1 in enumerate(experiment_names):
            for exp2 in experiment_names[i+1:]:
                if exp1 not in trials_data or exp2 not in trials_data:
                    continue
                    
                exp1_trials = trials_data[exp1]
                exp2_trials = trials_data[exp2]
                comparison_key = f"{exp1}_vs_{exp2}"
                epoch_wise_significance[comparison_key] = {}
                
                for epoch_idx in range(n_epochs):
                    exp1_vals = exp1_trials[:, epoch_idx]
                    exp2_vals = exp2_trials[:, epoch_idx]
                    
                    t_stat, p_val = stats.ttest_ind(exp1_vals, exp2_vals, equal_var=False)
                    
                    epoch_wise_significance[comparison_key][epoch_idx] = {
                        'p_value': p_val,
                        't_statistic': t_stat,
                        'significant': p_val < 0.05,
                        'mean_diff': np.mean(exp1_vals) - np.mean(exp2_vals)
                    }
    
    return epoch_wise_significance, final_values

def plot_results(stats_data, epoch_wise_significance, final_values, metrics_data, output_path='surrogate_comparison.png', 
                 baseline_name=None, title_suffix=''):
    """
    Create a publication-quality plot with confidence intervals and significance markers.
    """
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
    
    # Color palette
    colors = sns.color_palette("husl", len(stats_data))
    
    # Plot 1: Learning curves with confidence intervals and significance markers
    exp_names = list(stats_data.keys())
    
    for (exp_name, data), color in zip(stats_data.items(), colors):
        epochs = data['epochs']
        mean = data['mean']
        ci_lower = data['ci_lower']
        ci_upper = data['ci_upper']
        n_trials = data['n_trials']
        
        # Create friendly display name
        metadata = metrics_data.get(exp_name, {}).get('metadata', {})
        display_name = create_display_name(exp_name, metadata)
        is_llm = metadata.get('uses_llm', False)
        
        label = f"{display_name} (n={n_trials})"
        ax1.plot(epochs, mean, label=label, linewidth=2, color=color)
        ax1.fill_between(epochs, ci_lower, ci_upper, alpha=0.2, color=color)
        
        # Add stars where LLM is significantly better than baseline
        if is_llm and baseline_name and exp_name != baseline_name:
            comparison_key = f"{exp_name}_vs_{baseline_name}"
            if comparison_key in epoch_wise_significance:
                # Find epochs where LLM is significantly better (positive mean_diff and p < 0.05)
                for epoch_idx, sig_data in epoch_wise_significance[comparison_key].items():
                    if sig_data['significant'] and sig_data['mean_diff'] > 0:
                        # Add a star at this epoch
                        ax1.plot(epochs[epoch_idx], mean[epoch_idx], marker='*', 
                                markersize=12, color=color, markeredgecolor='black', 
                                markeredgewidth=0.5, zorder=10)
    
    ax1.set_xlabel('Search Iterations (Model Updates)')
    ax1.set_ylabel('Kendall Tau Correlation')
    ax1.set_title(f'Surrogate Model Performance Over Time{title_suffix}')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance comparison with significance
    display_names = [create_display_name(name, metrics_data.get(name, {}).get('metadata', {})) 
                     for name in exp_names]
    final_means = [stats_data[name]['mean'][-1] for name in exp_names]
    final_sems = [stats_data[name]['sem'][-1] for name in exp_names]
    
    x_pos = np.arange(len(exp_names))
    # bars = ax2.bar(x_pos, final_means, yerr=final_sems, capsize=5, 
    #                color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add significance stars for final epoch
    # if baseline_name and baseline_name in exp_names:
    #     baseline_idx = exp_names.index(baseline_name)
    #     baseline_mean = final_means[baseline_idx]
    #     y_max = max(final_means) + max(final_sems)

    #     for i, exp_name in enumerate(exp_names):
    #         if exp_name == baseline_name:
    #             continue
            
    #         comparison_key = f"{exp_name}_vs_{baseline_name}"
    #         if comparison_key in epoch_wise_significance:
    #             # Get the last epoch index
    #             last_epoch_idx = max(epoch_wise_significance[comparison_key].keys())
    #             final_sig = epoch_wise_significance[comparison_key][last_epoch_idx]
                
    #             if final_sig['significant']:
    #                 # Add star above bar
    #                 y_pos = final_means[i] + final_sems[i] + 0.02
    #                 ax2.text(i, y_pos, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def print_summary_statistics(stats_data, p_values, final_values, metrics_data=None):
    """
    Print a formatted summary of results.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for exp_name, data in stats_data.items():
        print(f"\n{exp_name}:")
        
        # Print metadata if available
        if metrics_data and exp_name in metrics_data:
            metadata = metrics_data[exp_name].get('metadata', {})
            if metadata:
                print(f"  Predictor: {metadata.get('predictor_class', 'N/A')}")
                if metadata.get('uses_llm'):
                    print(f"  Base Surrogate: {metadata.get('base_predictor', 'N/A')}")
                print(f"  Surrogate Type: {metadata.get('surrogate_type', 'N/A')}")
        
        print(f"  Number of trials: {data['n_trials']}")
        print(f"  Final Kendall Tau: {data['mean'][-1]:.4f} ± {data['sem'][-1]:.4f}")
        print(f"  95% CI: [{data['ci_lower'][-1]:.4f}, {data['ci_upper'][-1]:.4f}]")
        if exp_name in final_values:
            print(f"  Min/Max across trials: [{np.min(final_values[exp_name]):.4f}, {np.max(final_values[exp_name]):.4f}]")
    
    print("\n" + "="*80)
    print("SIGNIFICANCE TESTS (Epoch-wise Summary)")
    print("="*80)
    
    for comparison, epoch_results in p_values.items():
        if not epoch_results:
            continue
            
        # Get final epoch result
        final_epoch_idx = max(epoch_results.keys())
        final_result = epoch_results[final_epoch_idx]
        
        # Count significant epochs
        sig_epochs = [idx for idx, res in epoch_results.items() if res['significant']]
        total_epochs = len(epoch_results)
        
        sig_marker = "***" if final_result['p_value'] < 0.001 else "**" if final_result['p_value'] < 0.01 else "*" if final_result['significant'] else ""
        print(f"\n{comparison}:")
        print(f"  Final epoch mean difference: {final_result['mean_diff']:+.4f}")
        print(f"  Final epoch t-statistic: {final_result['t_statistic']:.4f}")
        print(f"  Final epoch p-value: {final_result['p_value']:.4f} {sig_marker}")
        print(f"  Significant at final epoch (α=0.05): {final_result['significant']}")
        print(f"  Significant epochs: {len(sig_epochs)}/{total_epochs} ({100*len(sig_epochs)/total_epochs:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze surrogate model performance')
    parser.add_argument('--results_dir', type=str, default='run_nb201',
                        help='Directory containing experiment results')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset to analyze (cifar10, cifar100, ImageNet16-120)')
    parser.add_argument('--baseline', type=str, default='CustomXGBoost_bananas',
                        help='Baseline experiment name for significance testing')
    parser.add_argument('--output', type=str, default='surrogate_comparison.png',
                        help='Output plot filename')
    
    args = parser.parse_args()
    
    print(f"Loading metrics from {args.results_dir} for dataset {args.dataset}...")
    
    # Load all metrics
    metrics_data = load_metrics(args.results_dir, args.dataset)
    
    if not metrics_data:
        print("No metrics found! Make sure you've run experiments and saved metrics.")
        return
    
    print(f"\nFound experiments: {list(metrics_data.keys())}")
    
    # Compute statistics
    stats_data = compute_statistics(metrics_data)
    
    # Perform epoch-wise significance tests
    epoch_wise_significance, final_values = perform_significance_tests_per_epoch(metrics_data, baseline_name=args.baseline)
    
    # Print summary
    print_summary_statistics(stats_data, epoch_wise_significance, final_values, metrics_data)
    
    # Create plots
    plot_results(stats_data, epoch_wise_significance, final_values, metrics_data,
                 output_path=args.output,
                 baseline_name=args.baseline,
                 title_suffix=f' ({args.dataset})')
    
    print(f"\nAnalysis complete!")

if __name__ == '__main__':
    main()
