import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Update these paths to point to your actual 3 json files
space = "nasbench101"
start_seed = 1242
num_submissions = 5
seeds = [start_seed + i for i in range(num_submissions)]
trials = 10
all_seeds = []
for seed in seeds:
    for trial in range(trials):
        all_seeds.append(seed * (trial+1))
all_seeds = list(set(all_seeds))
dataset = "cifar10"
surrogate = "xgboost"

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
    ylims = (90, 96)
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

output_filename = f"results/search_trajectory_comparison_{space}_{dataset}_avg_{len(all_seeds)}seeds_startseed_{start_seed}.png"

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
    
    return x_vals, y_mean, y_std

# --- Plotting ---
plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
markers = ['o', 's', '^']

print(f"\nProcessing {len(all_seeds)} seeds: {all_seeds}\n")

# Store mean values for pairwise comparison
method_means = {}

for i, (label, file_template) in enumerate(file_paths.items()):
    result = load_all_seeds(file_template, all_seeds)
    
    if result[0] is not None:
        x, y_mean, y_std = result
        
        # Store for pairwise comparison
        method_means[label] = y_mean
        
        # Plot mean line
        plt.plot(x, y_mean, label=label, color=colors[i], marker=markers[i], 
                 markersize=5, markevery=max(1, len(x)//20), linewidth=2, alpha=0.8)
        
        # Plot shaded std region
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        color=colors[i], alpha=0.2)

# Calculate and print pairwise differences
if len(method_means) >= 2:
    print("\n=== Average Pairwise Differences (% Validation Accuracy) ===")
    methods = list(method_means.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            max_common_length = min(len(method_means[method1]), len(method_means[method2]))
            diff = np.mean(method_means[method2] - method_means[method1])
            print(f"Cumulative: {method2} - {method1}: {diff:+.3f}%")
            diff = method_means[method2][max_common_length-1] - method_means[method1][max_common_length-1]
            print(f"Best individual: {method2} - {method1}: {diff:+.3f}%")
    print()

# Styling
plt.title(f"NAS Search Trajectory: {dataset.upper()} ({space}) - Average of {len(all_seeds)} Seeds", fontsize=14)
plt.xlabel("Number of Architectures Evaluated (Queries)", fontsize=12)
plt.ylabel("Validation Accuracy (%)", fontsize=12)
# hardcode y lower and upper limit
plt.ylim(ylims[0], ylims[1])

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

# Save
os.makedirs("results", exist_ok=True)
plt.savefig(output_filename, dpi=300)
print(f"\nPlot saved to {output_filename}")