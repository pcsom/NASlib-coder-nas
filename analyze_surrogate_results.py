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

def calculate_all_seeds(start_seed, num_submissions, trials):
    """
    Calculate all seed values based on start_seed, num_submissions, and trials.
    Uses the same logic as results_average.py.
    
    Args:
        start_seed: Starting seed value
        num_submissions: Number of sequential base seeds
        trials: Number of trials per base seed
    
    Returns:
        list: All unique seed values
    """
    seeds = [start_seed + i for i in range(num_submissions)]
    all_seeds = []
    for seed in seeds:
        for trial in range(trials):
            all_seeds.append(seed * (trial+1))
    all_seeds = list(set(all_seeds))
    return all_seeds

def load_metrics(results_dir, dataset='cifar10', all_seeds=None):
    """
    Load all surrogate metrics from JSON files.
    
    Args:
        results_dir: Directory containing results
        dataset: Dataset name
        all_seeds: Optional list of seed values to filter by. If None, loads all.
    
    Returns:
        tuple: (kendall_tau_metrics, mse_metrics) where each is
               dict: {experiment_id: {'epochs': [...], 'trials': [...], 'metadata': {...}}}
    """
    metrics_tau = defaultdict(lambda: {'epochs': None, 'trials': [], 'metadata': {}})
    metrics_mse = defaultdict(lambda: {'epochs': None, 'trials': [], 'metadata': {}})
    
    # Find all JSON metric files
    pattern = os.path.join(results_dir, 'nasbench201', dataset, '**', 'surrogate_metrics_*.json')
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} total metric files")
    
    # Filter files by seed if all_seeds is provided
    if all_seeds is not None:
        filtered_files = []
        seeds_found = set()
        for filepath in files:
            # Extract seed from path (format: .../seed/surrogate_metrics_*.json)
            parts = filepath.split(os.sep)
            # Find the seed directory (should be a numeric folder name)
            for part in parts:
                if part.isdigit() and int(part) in all_seeds:
                    filtered_files.append(filepath)
                    seeds_found.add(int(part))
                    break
        files = filtered_files
        print(f"Filtered to {len(files)} files matching specified seeds")
        print(f"Seeds requested: {len(all_seeds)}, Seeds found: {len(seeds_found)}")
        
        # Report missing seeds
        missing_seeds = set(all_seeds) - seeds_found
        if missing_seeds:
            print(f"Warning: Missing data for seeds: {sorted(missing_seeds)}")
        
        # Report actual seeds found
        print(f"Actual seeds with data: {sorted(seeds_found)}")
    
    seeds_per_experiment = defaultdict(set)  # Track which seeds each experiment has
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Use experiment_id if available, fallback to old 'optimizer' field
            experiment_id = data.get('experiment_id', data.get('optimizer', 'unknown'))
            epochs = data['epochs']
            kendall_tau = data['kendall_tau']
            mse = data.get('mse', None)  # MSE might not exist in older files
            
            # Track which seed this is for
            if all_seeds is not None:
                parts = filepath.split(os.sep)
                for part in parts:
                    if part.isdigit() and int(part) in all_seeds:
                        seeds_per_experiment[experiment_id].add(int(part))
                        break
            
            # Store metadata from first trial (shared between tau and mse)
            metadata_dict = {
                'predictor_class': data.get('predictor_class', 'unknown'),
                'optimizer_type': data.get('optimizer_type', 'unknown'),
                'surrogate_type': data.get('surrogate_type', 'unknown'),
                'uses_llm': data.get('uses_llm', False),
                'base_predictor': data.get('base_predictor', None),
                'dataset': data.get('dataset', dataset)
            }
            
            if not metrics_tau[experiment_id]['metadata']:
                metrics_tau[experiment_id]['metadata'] = metadata_dict
            if not metrics_mse[experiment_id]['metadata']:
                metrics_mse[experiment_id]['metadata'] = metadata_dict
            
            # Store epochs (should be the same for all trials)
            if metrics_tau[experiment_id]['epochs'] is None:
                metrics_tau[experiment_id]['epochs'] = epochs
            if metrics_mse[experiment_id]['epochs'] is None:
                metrics_mse[experiment_id]['epochs'] = epochs
            
            # Store trial data
            metrics_tau[experiment_id]['trials'].append(kendall_tau)
            if mse is not None:
                metrics_mse[experiment_id]['trials'].append(mse)
            
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    # Report per-experiment seed coverage
    if all_seeds is not None and seeds_per_experiment:
        print(f"\nPer-experiment seed coverage:")
        for exp_id, exp_seeds in sorted(seeds_per_experiment.items()):
            missing = set(all_seeds) - exp_seeds
            if missing:
                print(f"  {exp_id}: {len(exp_seeds)}/{len(all_seeds)} seeds (missing: {sorted(missing)})")
            else:
                print(f"  {exp_id}: {len(exp_seeds)}/{len(all_seeds)} seeds (complete)")
    
    return dict(metrics_tau), dict(metrics_mse)

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

def plot_results(stats_data_tau, epoch_wise_significance_tau, final_values_tau, metrics_data_tau,
                 stats_data_mse, epoch_wise_significance_mse, final_values_mse, metrics_data_mse,
                 output_path='surrogate_comparison.png', 
                 baseline_name=None, title_suffix='', k=10):
    """
    Create a publication-quality plot with confidence intervals and significance markers.
    Now includes 4 panels: 2 for Kendall Tau, 2 for MSE.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12), 
                                                   gridspec_kw={'height_ratios': [2, 2]})
    
    # Color palette
    colors = sns.color_palette("husl", len(stats_data_tau))
    
    # ========== PANEL 1: Kendall Tau Learning Curves ==========
    exp_names = list(stats_data_tau.keys())
    
    # First pass: plot all lines for Kendall Tau
    for (exp_name, data), color in zip(stats_data_tau.items(), colors):
        epochs = np.array(data['epochs']) * k  # Multiply by k for display (k architectures per epoch)
        mean = data['mean']
        ci_lower = data['ci_lower']
        ci_upper = data['ci_upper']
        n_trials = data['n_trials']
        
        # Create friendly display name
        metadata = metrics_data_tau.get(exp_name, {}).get('metadata', {})
        display_name = create_display_name(exp_name, metadata)
        is_llm = metadata.get('uses_llm', False)
        
        label = f"{display_name} (n={n_trials})"
        ax1.plot(epochs, mean, label=label, linewidth=2, color=color)
        ax1.fill_between(epochs, ci_lower, ci_upper, alpha=0.2, color=color)
    
    # Second pass: add significance markers for Kendall Tau
    for (exp_name, data), color in zip(stats_data_tau.items(), colors):
        epochs = np.array(data['epochs']) * k
        mean = data['mean']
        metadata = metrics_data_tau.get(exp_name, {}).get('metadata', {})
        is_llm = metadata.get('uses_llm', False)
        
        # Add stars where LLM is significantly better than baseline
        if is_llm and baseline_name and exp_name != baseline_name:
            comparison_key = f"{exp_name}_vs_{baseline_name}"
            if comparison_key in epoch_wise_significance_tau:
                # Find epochs where LLM is significantly better (positive mean_diff and p < 0.05)
                for epoch_idx, sig_data in epoch_wise_significance_tau[comparison_key].items():
                    if sig_data['significant'] and sig_data['mean_diff'] > 0:
                        # Add a star at this epoch on LLM line
                        ax1.plot(epochs[epoch_idx], mean[epoch_idx], marker='*', 
                                markersize=12, color=color, markeredgecolor='black', 
                                markeredgewidth=0.5, zorder=10)
        
        # Add stars where baseline is significantly better than LLM
        if exp_name == baseline_name and baseline_name:
            baseline_epochs = epochs
            baseline_mean = mean
            baseline_color = color
            
            # Check all LLM experiments
            for other_exp_name in exp_names:
                if other_exp_name == baseline_name:
                    continue
                other_metadata = metrics_data_tau.get(other_exp_name, {}).get('metadata', {})
                if other_metadata.get('uses_llm', False):
                    comparison_key = f"{other_exp_name}_vs_{baseline_name}"
                    if comparison_key in epoch_wise_significance_tau:
                        # Find epochs where baseline is significantly better (negative mean_diff and p < 0.05)
                        for epoch_idx, sig_data in epoch_wise_significance_tau[comparison_key].items():
                            if sig_data['significant'] and sig_data['mean_diff'] < 0:
                                # Add a star at this epoch on baseline line
                                ax1.plot(baseline_epochs[epoch_idx], baseline_mean[epoch_idx], marker='*', 
                                        markersize=12, color=baseline_color, markeredgecolor='black', 
                                        markeredgewidth=0.5, zorder=10)
    
    ax1.set_xlabel('Number of Architectures Evaluated (Queries)')
    ax1.set_ylabel("Kendall's Tau Correlation")
    ax1.set_title(f'Kendall Tau: Surrogate Model Performance{title_suffix}')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== PANEL 2: Kendall Tau P-values ==========
    if baseline_name and baseline_name in exp_names:
        # Find LLM experiments to compare against baseline
        llm_experiments = []
        for exp_name in exp_names:
            if exp_name != baseline_name:
                metadata = metrics_data_tau.get(exp_name, {}).get('metadata', {})
                if metadata.get('uses_llm', False):
                    llm_experiments.append(exp_name)
        
        if llm_experiments:
            for exp_name, color in zip(llm_experiments, colors[1:]):  # Skip first color (baseline)
                comparison_key = f"{exp_name}_vs_{baseline_name}"
                if comparison_key in epoch_wise_significance_tau:
                    epochs = np.array(stats_data_tau[exp_name]['epochs']) * k
                    p_values = [epoch_wise_significance_tau[comparison_key][i]['p_value'] 
                               for i in range(len(stats_data_tau[exp_name]['epochs']))]
                    
                    metadata = metrics_data_tau.get(exp_name, {}).get('metadata', {})
                    display_name = create_display_name(exp_name, metadata)
                    
                    ax2.plot(epochs, p_values, color=color, linewidth=2, 
                            label=f'{display_name} vs {create_display_name(baseline_name, metrics_data_tau.get(baseline_name, {}).get("metadata", {}))}')
            
            ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, 
                       label='p=0.05 (significance threshold)', alpha=0.7)
            ax2.axhline(y=0.01, color='darkred', linestyle=':', linewidth=1, 
                       label='p=0.01', alpha=0.7)
            ax2.set_xlabel('Number of Architectures Evaluated (Queries)')
            ax2.set_ylabel('P-value (t-test)')
            ax2.set_title('Kendall Tau: Statistical Significance Over Time')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)
    
    # ========== PANEL 3: MSE Learning Curves ==========
    exp_names_mse = list(stats_data_mse.keys())
    
    # First pass: plot all lines for MSE
    for (exp_name, data), color in zip(stats_data_mse.items(), colors):
        if data['n_trials'] == 0:
            continue
        epochs = np.array(data['epochs']) * k
        mean = data['mean']
        ci_lower = data['ci_lower']
        ci_upper = data['ci_upper']
        n_trials = data['n_trials']
        
        metadata = metrics_data_mse.get(exp_name, {}).get('metadata', {})
        display_name = create_display_name(exp_name, metadata)
        
        label = f"{display_name} (n={n_trials})"
        ax3.plot(epochs, mean, label=label, linewidth=2, color=color)
        ax3.fill_between(epochs, ci_lower, ci_upper, alpha=0.2, color=color)
    
    # Second pass: add significance markers for MSE
    for (exp_name, data), color in zip(stats_data_mse.items(), colors):
        if data['n_trials'] == 0:
            continue
        epochs = np.array(data['epochs']) * k
        mean = data['mean']
        metadata = metrics_data_mse.get(exp_name, {}).get('metadata', {})
        is_llm = metadata.get('uses_llm', False)
        
        # Add stars where LLM has significantly lower MSE (better)
        if is_llm and baseline_name and exp_name != baseline_name:
            comparison_key = f"{exp_name}_vs_{baseline_name}"
            if comparison_key in epoch_wise_significance_mse:
                for epoch_idx, sig_data in epoch_wise_significance_mse[comparison_key].items():
                    # For MSE, negative mean_diff means LLM is better (lower MSE)
                    if sig_data['significant'] and sig_data['mean_diff'] < 0:
                        ax3.plot(epochs[epoch_idx], mean[epoch_idx], marker='*', 
                                markersize=12, color=color, markeredgecolor='black', 
                                markeredgewidth=0.5, zorder=10)
        
        # Add stars where baseline has significantly lower MSE
        if exp_name == baseline_name and baseline_name:
            baseline_epochs = epochs
            baseline_mean = mean
            baseline_color = color
            
            for other_exp_name in exp_names_mse:
                if other_exp_name == baseline_name:
                    continue
                other_metadata = metrics_data_mse.get(other_exp_name, {}).get('metadata', {})
                if other_metadata.get('uses_llm', False):
                    comparison_key = f"{other_exp_name}_vs_{baseline_name}"
                    if comparison_key in epoch_wise_significance_mse:
                        for epoch_idx, sig_data in epoch_wise_significance_mse[comparison_key].items():
                            # For MSE, positive mean_diff means baseline is better
                            if sig_data['significant'] and sig_data['mean_diff'] > 0:
                                ax3.plot(baseline_epochs[epoch_idx], baseline_mean[epoch_idx], marker='*', 
                                        markersize=12, color=baseline_color, markeredgecolor='black', 
                                        markeredgewidth=0.5, zorder=10)
    
    ax3.set_xlabel('Number of Architectures Evaluated (Queries)')
    ax3.set_ylabel('Mean Squared Error (MSE)')
    ax3.set_title(f'MSE: Surrogate Model Performance{title_suffix}')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== PANEL 4: MSE P-values ==========
    if baseline_name and baseline_name in exp_names_mse:
        llm_experiments_mse = []
        for exp_name in exp_names_mse:
            if exp_name != baseline_name:
                metadata = metrics_data_mse.get(exp_name, {}).get('metadata', {})
                if metadata.get('uses_llm', False):
                    llm_experiments_mse.append(exp_name)
        
        if llm_experiments_mse:
            for exp_name, color in zip(llm_experiments_mse, colors[1:]):
                comparison_key = f"{exp_name}_vs_{baseline_name}"
                if comparison_key in epoch_wise_significance_mse:
                    if exp_name in stats_data_mse and stats_data_mse[exp_name]['n_trials'] > 0:
                        epochs = np.array(stats_data_mse[exp_name]['epochs']) * k
                        p_values = [epoch_wise_significance_mse[comparison_key][i]['p_value'] 
                                   for i in range(len(stats_data_mse[exp_name]['epochs']))]
                        
                        metadata = metrics_data_mse.get(exp_name, {}).get('metadata', {})
                        display_name = create_display_name(exp_name, metadata)
                        
                        ax4.plot(epochs, p_values, color=color, linewidth=2, 
                                label=f'{display_name} vs {create_display_name(baseline_name, metrics_data_mse.get(baseline_name, {}).get("metadata", {}))}')
            
            ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=1, 
                       label='p=0.05 (significance threshold)', alpha=0.7)
            ax4.axhline(y=0.01, color='darkred', linestyle=':', linewidth=1, 
                       label='p=0.01', alpha=0.7)
            ax4.set_xlabel('Number of Architectures Evaluated (Queries)')
            ax4.set_ylabel('P-value (t-test)')
            ax4.set_title('MSE: Statistical Significance Over Time')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def print_summary_statistics(stats_data_tau, p_values_tau, final_values_tau, metrics_data_tau,
                            stats_data_mse, p_values_mse, final_values_mse, metrics_data_mse):
    """
    Print a formatted summary of results for both Kendall Tau and MSE.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - KENDALL TAU")
    print("="*80)
    
    for exp_name, data in stats_data_tau.items():
        print(f"\n{exp_name}:")
        
        # Print metadata if available
        if metrics_data_tau and exp_name in metrics_data_tau:
            metadata = metrics_data_tau[exp_name].get('metadata', {})
            if metadata:
                print(f"  Predictor: {metadata.get('predictor_class', 'N/A')}")
                if metadata.get('uses_llm'):
                    print(f"  Base Surrogate: {metadata.get('base_predictor', 'N/A')}")
                print(f"  Surrogate Type: {metadata.get('surrogate_type', 'N/A')}")
        
        print(f"  Number of trials: {data['n_trials']}")
        print(f"  Final Kendall Tau: {data['mean'][-1]:.4f} ± {data['sem'][-1]:.4f}")
        print(f"  95% CI: [{data['ci_lower'][-1]:.4f}, {data['ci_upper'][-1]:.4f}]")
        if exp_name in final_values_tau:
            print(f"  Min/Max across trials: [{np.min(final_values_tau[exp_name]):.4f}, {np.max(final_values_tau[exp_name]):.4f}]")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - MSE")
    print("="*80)
    
    for exp_name, data in stats_data_mse.items():
        if data['n_trials'] == 0:
            continue
        print(f"\n{exp_name}:")
        print(f"  Number of trials: {data['n_trials']}")
        print(f"  Final MSE: {data['mean'][-1]:.6f} ± {data['sem'][-1]:.6f}")
        print(f"  95% CI: [{data['ci_lower'][-1]:.6f}, {data['ci_upper'][-1]:.6f}]")
        if exp_name in final_values_mse:
            print(f"  Min/Max across trials: [{np.min(final_values_mse[exp_name]):.6f}, {np.max(final_values_mse[exp_name]):.6f}]")
    
    print("\n" + "="*80)
    print("SIGNIFICANCE TESTS - KENDALL TAU (Epoch-wise Summary)")
    print("="*80)
    
    for comparison, epoch_results in p_values_tau.items():
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
    
    print("\n" + "="*80)
    print("SIGNIFICANCE TESTS - MSE (Epoch-wise Summary)")
    print("="*80)
    
    for comparison, epoch_results in p_values_mse.items():
        if not epoch_results:
            continue
            
        final_epoch_idx = max(epoch_results.keys())
        final_result = epoch_results[final_epoch_idx]
        
        sig_epochs = [idx for idx, res in epoch_results.items() if res['significant']]
        total_epochs = len(epoch_results)
        
        sig_marker = "***" if final_result['p_value'] < 0.001 else "**" if final_result['p_value'] < 0.01 else "*" if final_result['significant'] else ""
        print(f"\n{comparison}:")
        print(f"  Final epoch mean difference: {final_result['mean_diff']:+.6f}")
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
    parser.add_argument('--start_seed', type=int, default=None,
                        help='Starting seed value (if specified, filters results by seed)')
    parser.add_argument('--num_submissions', type=int, default=5,
                        help='Number of sequential base seeds')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials per base seed')
    parser.add_argument('--k', type=int, default=10,
                        help='Batch size (number of architectures evaluated per epoch)')
    
    args = parser.parse_args()
    
    # Calculate seeds if start_seed is provided
    all_seeds = None
    if args.start_seed is not None:
        all_seeds = calculate_all_seeds(args.start_seed, args.num_submissions, args.trials)
        print(f"Filtering results for seeds: {all_seeds}")
    
    print(f"Loading metrics from {args.results_dir} for dataset {args.dataset}...")
    
    # Load all metrics (returns both Kendall Tau and MSE)
    metrics_data_tau, metrics_data_mse = load_metrics(args.results_dir, args.dataset, all_seeds=all_seeds)
    
    if not metrics_data_tau:
        print("No metrics found! Make sure you've run experiments and saved metrics.")
        return
    
    print(f"\nFound experiments (Kendall Tau): {list(metrics_data_tau.keys())}")
    print(f"Found experiments (MSE): {list(metrics_data_mse.keys())}")
    
    # Compute statistics for both metrics
    stats_data_tau = compute_statistics(metrics_data_tau)
    stats_data_mse = compute_statistics(metrics_data_mse)
    
    # Perform epoch-wise significance tests for both metrics
    epoch_wise_significance_tau, final_values_tau = perform_significance_tests_per_epoch(metrics_data_tau, baseline_name=args.baseline)
    epoch_wise_significance_mse, final_values_mse = perform_significance_tests_per_epoch(metrics_data_mse, baseline_name=args.baseline)
    
    # Print summary
    print_summary_statistics(stats_data_tau, epoch_wise_significance_tau, final_values_tau, metrics_data_tau,
                            stats_data_mse, epoch_wise_significance_mse, final_values_mse, metrics_data_mse)
    
    # Create plots
    plot_results(stats_data_tau, epoch_wise_significance_tau, final_values_tau, metrics_data_tau,
                 stats_data_mse, epoch_wise_significance_mse, final_values_mse, metrics_data_mse,
                 output_path=args.output,
                 baseline_name=args.baseline,
                 title_suffix=f' ({args.dataset})',
                 k=args.k)
    
    print(f"\nAnalysis complete!")

if __name__ == '__main__':
    main()
