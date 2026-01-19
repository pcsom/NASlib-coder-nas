import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Update these paths to point to your actual 3 json files
space = "nasbench301"
seed = 1343
dataset = "cifar10"
surrogate = "xgboost"

surrogate_name_map = {
    "xgboost": "CustomXGBoost",
    "mlp": "CustomMLP"
}

if space == "nasbench201":
    llm_pred = f"LLM_NB201_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb201"
elif space == "nasbench301":
    llm_pred = f"LLM_NB301_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb301"
elif space == "nasbench101":
    llm_pred = f"LLM_NB101_Predictor_{surrogate_name_map[surrogate]}_bananas"
    run = "run_nb101"

surrogate_folder = f"{surrogate_name_map[surrogate]}_bananas"

file_paths = {
    "REA (Default)": f"/home/hice1/psomu3/scratch/codenas/NASLib/{run}/{space}/{dataset}/Default_rea/{seed}/errors.json",
    "Bananas (Default)": f"/home/hice1/psomu3/scratch/codenas/NASLib/{run}/{space}/{dataset}/{surrogate_folder}/{seed}/errors.json",
    "Bananas + LLM (Ours)": f"/home/hice1/psomu3/scratch/codenas/NASLib/{run}/{space}/{dataset}/{llm_pred}/{seed}/errors.json"
}

output_filename = "results/search_trajectory_comparison"
output_filename += f"_{space}_{dataset}_{seed}.png"

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
    
    # Calculate X-axis (Number of Architectures Evaluated)
    # Queries = iteration_index
    num_init = config.get('num_init', 10)
    k = config.get('k', 10) # Batch size per step
    
    x_vals = range(len(y_vals))
    
    return x_vals, y_vals

# --- Plotting ---
plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
markers = ['o', 's', '^']

for i, (label, filepath) in enumerate(file_paths.items()):
    x, y = load_data(filepath)
    
    if x is not None and y is not None:
        plt.plot(x, y, label=label, color=colors[i], marker=markers[i], 
                 markersize=5, markevery=5, linewidth=2, alpha=0.8)

# Styling
plt.title(f"NAS Search Trajectory: CIFAR-10 ({space})", fontsize=14)
plt.xlabel("Number of Architectures Evaluated (Queries)", fontsize=12)
plt.ylabel("Validation Accuracy (%)", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

# Save
plt.savefig(output_filename, dpi=300)
print(f"Plot saved to {output_filename}")