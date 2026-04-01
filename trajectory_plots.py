import json
import numpy as np
import pandas as pd

import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns


def surrogate_data_plots(res_dir):
    #load res_dir/candiate_log_trail_0_seed_*.json where * can be anything you  want t
    files = glob.glob(os.path.join(res_dir, 'candidate_log_trial_0_seed_*.json'))
    data = []
    for file in files:
        with open(file, 'r') as f:
            #retrieve generation, predicted accuracy, true accuracy for each individual (and this json file is a list of individuals)
            trial_data = json.load(f)
            for entry in trial_data:
                data.append({
                    'epoch': entry['generation'],
                    'true_accuracy': entry['true_accuracy'],
                    'predicted_accuracy': entry['predicted_accuracy']
                })
        #create a plot of predicted vs true accuracy across generations (x-axis is generation, y-axis is accuracy error) and save it as res_dir/surrogate_accuracy_by_gen.png
        df = pd.DataFrame(data)
        df['accuracy_error'] = np.abs(df['predicted_accuracy'] - df['true_accuracy'])
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='accuracy_error', data=df)
        plt.xlabel('Generation')
        plt.ylabel('Absolute Accuracy Error')
        plt.title('Surrogate Accuracy Error Across Generations')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(df['epoch'], df['accuracy_error'], 1)
        plt.plot(df['epoch'], slope * df['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()

        plt.savefig(os.path.join(res_dir, 'surrogate_accuracy_by_gen.png'))
        plt.close()
        #create 2 plots of preductaed and true accuracy across generations (x-axis is generation, y-axis is accuracy) and save it as res_dir/surrogate_pred_true_by_gen.png
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='predicted_accuracy', data=df, label='True Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Average Predicted Accuracy')
        plt.title('Surrogate Predicted Accuracy Across Generations')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(df['epoch'], df['predicted_accuracy'], 1)
        plt.plot(df['epoch'], slope * df['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()
        plt.savefig(os.path.join(res_dir, 'surrogate_predict_accuracy_by_gen.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='epoch', y='true_accuracy', data=df, label='True Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Average True Accuracy')
        plt.title('Surrogate True Accuracy Across Generations')
        plt.grid(True)
        #draw line of best fit and also write the equation in legend
        slope, intercept = np.polyfit(df['epoch'], df['true_accuracy'], 1)
        plt.plot(df['epoch'], slope * df['epoch'] + intercept, color='red', label=f'Best Fit Line: y={slope:.6f}x + {intercept:.6f}')
        plt.legend()
        plt.savefig(os.path.join(res_dir, 'surrogate_true_accuracy_by_gen.png'))
        plt.close()