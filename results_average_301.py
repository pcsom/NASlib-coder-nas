import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# --- Configuration ---
# Update these paths to point to your actual 3 json files
space = "nasbench301"
start_seed = 2342
num_submissions = 4
seeds = [start_seed + i for i in range(num_submissions)]
trials = 1
all_seeds = []
for seed in seeds:
    for trial in range(trials):
        all_seeds.append(seed * (trial+1))
all_seeds = list(set(all_seeds))
dataset = "cifar10"
surrogate = "mlp"

surrogate_name_map = {
    "xgboost": "CustomXGBoost",
    "mlp": "CustomMLP"
}

if space == "nasbench201":
    llm_pred = f"LLM_NB201_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb201"
    if dataset == "cifar10":
        ylims = (82, 93)
    else:
        ylims = (60, 75)
elif space == "nasbench301":
    llm_pred = f"LLM_NB301_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb301"
    ylims = (92, 95)
elif space == "nasbench101":
    llm_pred = f"LLM_NB101_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb101"
    ylims = (87, 96)

surrogate_folder = f"{surrogate_name_map[surrogate]}_bananas"

file_paths = {
    "REA (Default)": f"/home/hice1/psomu3/scratch/codenas/NASLib/{run}/{space}/{dataset}/Default_rea/{{seed}}/errors.json",
    "Bananas (Default)": f"/home/hice1/psomu3/scratch/codenas/NASLib/{run}/{space}/{dataset}/{surrogate_folder}/{{seed}}/errors.json",
    "Bananas + LLM (Ours)": f"/home/hice1/psomu3/scratch/codenas/NASLib/{run}/{space}/{dataset}/{llm_pred}/{{seed}}/errors.json"
}

# Will update this with actual seeds loaded after loading data
output_filename_template = f"results/search_trajectory_comparison_{space}_{dataset}_avg_{{n_seeds}}seeds_startseed_{start_seed}.png"

def load_data(filepath):
    """Parses the NASLib results JSON."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return None, None
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # The JSON is a list: [config_dict, results_dict]
    config = data[0]['search']
    results = data[1]
    
    # Extract Metric (Test Accuracy of the Incumbent)
    # Using test_acc to see true performance, or valid_acc if you strictly want search progression
    y_vals = results['valid_acc']
    
    # Convert to "best so far" (cumulative maximum)
    y_vals = np.maximum.accumulate(y_vals)
    
    # Calculate X-axis (Number of Architectures Evaluated)
    # Queries = iteration_index
    num_init = config.get('num_init', 10)
    k = config.get('k', 10) # Batch size per step
    
    x_vals = range(len(y_vals))
    
    return x_vals, y_vals

def load_all_seeds(file_template, seeds):
    """Load data from multiple seeds and return aggregated results."""
    all_y_vals = []
    max_length = 0
    
    for seed in seeds:
        filepath = file_template.format(seed=seed)
        x, y = load_data(filepath)
        
        if x is not None and y is not None:
            all_y_vals.append(y)
            max_length = max(max_length, len(y))
    
    if not all_y_vals:
        print(f"Warning: No valid data found for template {file_template}")
        return None, None
    
    # Pad shorter sequences with their last value
    padded_y_vals = []
    for y in all_y_vals:
        if len(y) < max_length:
            padded = list(y) + [y[-1]] * (max_length - len(y))
        else:
            padded = y
        padded_y_vals.append(padded)
    
    # Calculate mean and std across seeds
    y_mean = np.mean(padded_y_vals, axis=0)
    y_std = np.std(padded_y_vals, axis=0)
    x_vals = range(len(y_mean))
    
    print(f"Loaded {len(all_y_vals)} seeds for template {file_template}")
    
    return x_vals, y_mean, y_std, np.array(padded_y_vals), len(all_y_vals)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
markers = ['o', 's', '^']

print(f"\nProcessing {len(all_seeds)} seeds: {all_seeds}\n")

# Store mean, std, and raw values for comparison and t-tests
method_means = {}
method_stds = {}
method_raw_data = {}
actual_seeds_loaded = 0  # Track actual number of seeds successfully loaded

for i, (label, file_template) in enumerate(file_paths.items()):
    result = load_all_seeds(file_template, all_seeds)
    
    if result[0] is not None:
        x, y_mean, y_std, y_raw, n_seeds_loaded = result
        
        # Update actual seeds count (use max across methods)
        actual_seeds_loaded = max(actual_seeds_loaded, n_seeds_loaded)
        
        # Store for comparison
        method_means[label] = y_mean
        method_stds[label] = y_std
        method_raw_data[label] = y_raw
        
        # Plot mean line
        ax1.plot(x, y_mean, label=label, color=colors[i], marker=markers[i], 
                 markersize=5, markevery=max(1, len(x)//20), linewidth=2, alpha=0.8)
        
        # Plot shaded std region
        ax1.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        color=colors[i], alpha=0.2)

# Set output filename with actual seeds loaded
output_filename = output_filename_template.format(n_seeds=actual_seeds_loaded)

# Add optimal architecture line for NASBench201
if space == "nasbench201":
    if dataset == "cifar100":
        optimal_acc = 73.49
    elif dataset == "cifar10":
        optimal_acc = 91.61
    else:
        optimal_acc = None
elif space == "nasbench301":
    optimal_acc = 95.00
    
if optimal_acc is not None:
    ax1.axhline(y=optimal_acc, color='black', linestyle='--', linewidth=1.5, 
                label=f'Optimal Architecture ({optimal_acc}%)', alpha=0.7)

        
# Calculate and print statistics
if len(method_means) >= 1:
    methods = list(method_means.keys())
    
    # Prepare text file path
    txt_filename = output_filename.replace('.png', '.txt')
    
    # Open file for writing
    with open(txt_filename, 'w') as f:        
        # Write optimal performance
        if optimal_acc is not None:
            f.write(f"=== Optimal Architecture Performance ===\n")
            f.write(f"Optimal: {optimal_acc:.2f}%\n\n")
            print(f"\n=== Optimal Architecture Performance ===")
            print(f"Optimal: {optimal_acc:.2f}%\n")
        
        # Write method statistics (best individual = last data point)
        f.write("=== Method Statistics - Best Individual (% Validation Accuracy) ===\n")
        print("=== Method Statistics - Best Individual (% Validation Accuracy) ===")
        
        for method in methods:
            mean_acc = method_means[method][-1]  # Mean of final point across seeds
            std_acc = method_stds[method][-1]     # Std of final point across seeds
            
            # Print to console
            print(f"{method}: Mean = {mean_acc:.3f}%, Std = {std_acc:.3f}%")
            
            # Write to file
            f.write(f"{method}: Mean = {mean_acc:.3f}%, Std = {std_acc:.3f}%\n")

        print()
        f.write("\n")

        f.write("=== Method Statistics - Cumulative (% Validation Accuracy) ===\n")
        print("=== Method Statistics - Cumulative (% Validation Accuracy) ===")
        for method in methods:
            cumulative_mean = sum(method_means[method][:300])/300
            print(f"{method}: Mean = {cumulative_mean:.3f}%")

        
        print()
        f.write("\n")
        
        # Calculate earliest iteration to reach threshold
        # Threshold = lowest final accuracy across all methods, rounded down to nearest 0.01
        final_accs = [method_means[method][-1] for method in methods]
        min_final_acc = min(final_accs)
        # threshold = np.floor(min_final_acc * 100) / 100  # Round down to nearest 0.01
        # threshold = min_final_acc - 0.05
        threshold = optimal_acc * 0.995
        
        f.write(f"=== Convergence Speed (Iterations to reach {threshold:.2f}%) ===\n")
        print(f"=== Convergence Speed (Iterations to reach {threshold:.2f}%) ===")
        
        for method in methods:
            # Find first iteration where method reaches threshold
            iterations_to_threshold = np.where(method_means[method] >= threshold)[0]
            if len(iterations_to_threshold) > 0:
                first_iter = iterations_to_threshold[0]
                print(f"{method}: {first_iter} iterations")
                f.write(f"{method}: {first_iter} iterations\n")
            else:
                print(f"{method}: Never reached threshold")
                f.write(f"{method}: Never reached threshold\n")
        
        print()
        f.write("\n")
        
        # Calculate and write pairwise differences
        if len(method_means) >= 2:
            f.write("=== Average Pairwise Differences (% Validation Accuracy) ===\n")
            print("=== Average Pairwise Differences (% Validation Accuracy) ===")
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    max_common_length = min(len(method_means[method1]), len(method_means[method2]))
                    diff_cumulative = np.mean(method_means[method2] - method_means[method1])
                    diff_best = method_means[method2][max_common_length-1] - method_means[method1][max_common_length-1]
                    
                    # Print to console
                    print(f"Cumulative: {method2} - {method1}: {diff_cumulative:+.3f}%")
                    print(f"Best individual: {method2} - {method1}: {diff_best:+.3f}%")
                    
                    # Write to file
                    f.write(f"Cumulative: {method2} - {method1}: {diff_cumulative:+.3f}%\n")
                    f.write(f"Best individual: {method2} - {method1}: {diff_best:+.3f}%\n")
    
    print(f"\nStatistics saved to {txt_filename}")
    print()

# Compute t-test between Bananas and Bananas + LLM at each generation
if "Bananas (Default)" in method_raw_data and "Bananas + LLM (Ours)" in method_raw_data:
    bananas_data = method_raw_data["Bananas (Default)"]
    bananas_llm_data = method_raw_data["Bananas + LLM (Ours)"]
    
    # Compute t-test at each iteration
    p_values = []
    num_iterations = bananas_data.shape[1]
    
    for iter_idx in range(num_iterations):
        # Get data for this iteration across all seeds
        bananas_scores = bananas_data[:, iter_idx]
        llm_scores = bananas_llm_data[:, iter_idx]
        
        # Perform two-sample t-test
        t_stat, p_val = stats.ttest_ind(llm_scores, bananas_scores)
        p_values.append(p_val)
    
    # Plot p-values
    ax2.plot(range(num_iterations), p_values, color='purple', linewidth=2, label='P-value')
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='p=0.05 (significance threshold)', alpha=0.7)
    ax2.axhline(y=0.01, color='darkred', linestyle=':', linewidth=1, label='p=0.01', alpha=0.7)
    ax2.set_xlabel('Number of Architectures Evaluated (Queries)', fontsize=12)
    ax2.set_ylabel('P-value (t-test)', fontsize=12)
    ax2.set_title('Statistical Significance: Bananas + LLM vs Bananas', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)
    
    print(f"\nT-test results:")
    print(f"  P-value at iteration 50: {p_values[min(49, len(p_values)-1)]:.4e}")
    print(f"  P-value at iteration 100: {p_values[min(99, len(p_values)-1)]:.4e}")
    print(f"  P-value at final iteration: {p_values[-1]:.4e}")
    print(f"  Number of iterations with p<0.05: {sum(1 for p in p_values if p < 0.05)}/{len(p_values)}")
    print()

# Styling for main plot
ax1.set_title(f"NAS Search Trajectory: {dataset.upper()} ({space}) - Average of {actual_seeds_loaded} Seeds", fontsize=14)
ax1.set_xlabel("Number of Architectures Evaluated (Queries)", fontsize=12)
ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
# hardcode y lower and upper limit
ax1.set_ylim(ylims[0], ylims[1])

ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)
plt.tight_layout()

# Save
os.makedirs("results", exist_ok=True)
fig.savefig(output_filename, dpi=300)
print(f"\nPlot saved to {output_filename}")