#read every json file in run_nb201/nasbench201/cifar-10/LLM_NB201_Predictor_CustomMLP_bananas

import json
import os
import numpy as np
import pandas as pd
# Define the path to the directory containing the JSON files
directory_path10 = 'run_nb201/nasbench201/cifar10/LLM_NB201_Predictor_CustomMLP_bananas'
directory_path100 = 'run_nb201/nasbench201/cifar100/LLM_NB201_Predictor_CustomMLP_bananas'
# Initialize an empty list to store the data
data_list10 = []
data_list100 = []
# Iterate through each file  and direcory in the directory
for root, dirs, files in os.walk(directory_path10):
    for filename in files:
        if filename.endswith(".json") and filename.startswith("candidate_log"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_list10.append(data)
for root, dirs, files in os.walk(directory_path100):
    for filename in files:
        if filename.endswith(".json") and filename.startswith("candidate_log"):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_list100.append(data)
print(f"Total number of JSON files read: {len(data_list10) + len(data_list100)}")
#create a dictionary with the following structure: "arch_index": {"predicted_accuracy" "generationfirstdiscovery", "generation_lastdiscovery"} match predicted accuracy to the entry with last discovery and first discovery
arch_dict_10 = {}
for data in data_list10:
    for item in data:
        arch_index = item['arch_index']
        predicted_accuracy = item['predicted_accuracy']
        generation_firstdiscovery = item['generation']
        generation_lastdiscovery = item['generation']
        if arch_index not in arch_dict_10:
            arch_dict_10[arch_index] = {"predicted_accuracy": predicted_accuracy, "generation_firstdiscovery": generation_firstdiscovery, "generation_lastdiscovery": generation_lastdiscovery, "showed_up": 1, "average_predicted_accuracy": predicted_accuracy, "average_generation": generation_firstdiscovery}
        else:
            arch_dict_10[arch_index]["showed_up"] += 1
            arch_dict_10[arch_index]["average_predicted_accuracy"] = (arch_dict_10[arch_index]["average_predicted_accuracy"] * (arch_dict_10[arch_index]["showed_up"] - 1) + predicted_accuracy) / arch_dict_10[arch_index]["showed_up"]
            arch_dict_10[arch_index]["average_generation"] = (arch_dict_10[arch_index]["average_generation"] * (arch_dict_10[arch_index]["showed_up"] - 1) + generation_firstdiscovery) / arch_dict_10[arch_index]["showed_up"]
            if generation_firstdiscovery < arch_dict_10[arch_index]["generation_firstdiscovery"]:
                arch_dict_10[arch_index]["generation_firstdiscovery"] = generation_firstdiscovery
            if generation_lastdiscovery > arch_dict_10[arch_index]["generation_lastdiscovery"]:
                arch_dict_10[arch_index]["generation_lastdiscovery"] = generation_lastdiscovery
                arch_dict_10[arch_index]["predicted_accuracy"] = predicted_accuracy
    
    #convert to pandas dataframe
arch_dict_100 = {}
for data in data_list100:
    for item in data:
        arch_index = item['arch_index']
        predicted_accuracy = item['predicted_accuracy']
        generation_firstdiscovery = item['generation']
        generation_lastdiscovery = item['generation']
        if arch_index not in arch_dict_100:
            arch_dict_100[arch_index] = {"predicted_accuracy": predicted_accuracy, "generation_firstdiscovery": generation_firstdiscovery, "generation_lastdiscovery": generation_lastdiscovery, "showed_up": 1, "average_predicted_accuracy": predicted_accuracy, "average_generation": generation_firstdiscovery}
        else:
            arch_dict_100[arch_index]["showed_up"] += 1
            arch_dict_100[arch_index]["average_predicted_accuracy"] = (arch_dict_100[arch_index]["average_predicted_accuracy"] * (arch_dict_100[arch_index]["showed_up"] - 1) + predicted_accuracy) / arch_dict_100[arch_index]["showed_up"]
            arch_dict_100[arch_index]["average_generation"] = (arch_dict_100[arch_index]["average_generation"] * (arch_dict_100[arch_index]["showed_up"] - 1) + generation_firstdiscovery) / arch_dict_100[arch_index]["showed_up"]
            if generation_firstdiscovery < arch_dict_100[arch_index]["generation_firstdiscovery"]:
                arch_dict_100[arch_index]["generation_firstdiscovery"] = generation_firstdiscovery
            if generation_lastdiscovery > arch_dict_100[arch_index]["generation_lastdiscovery"]:
                arch_dict_100[arch_index]["generation_lastdiscovery"] = generation_lastdiscovery
                arch_dict_100[arch_index]["predicted_accuracy"] = predicted_accuracy

#now join the two dictionaries on the arch_index and create a pandas dataframe with the following columns: "arch_index", "predicted_accuracy_cifar10", "generation_firstdiscovery_cifar10", "generation_lastdiscovery_cifar10", "predicted_accuracy_cifar100", "generation_firstdiscovery_cifar100", "generation_lastdiscovery_cifar100"
data_combined = []
for arch_index in arch_dict_10:
    if arch_index in arch_dict_100:
        data_combined.append({
            "arch_index": arch_index,
            "predicted_accuracy_cifar10": arch_dict_10[arch_index]["predicted_accuracy"],
            "generation_firstdiscovery_cifar10": arch_dict_10[arch_index]["generation_firstdiscovery"],
            "generation_lastdiscovery_cifar10": arch_dict_10[arch_index]["generation_lastdiscovery"],
            "predicted_accuracy_cifar100": arch_dict_100[arch_index]["predicted_accuracy"],
            "generation_firstdiscovery_cifar100": arch_dict_100[arch_index]["generation_firstdiscovery"],
            "generation_lastdiscovery_cifar100": arch_dict_100[arch_index]["generation_lastdiscovery"],
            "average_predicted_accuracy_cifar10": arch_dict_10[arch_index]["average_predicted_accuracy"],
            "average_generation_cifar10": arch_dict_10[arch_index]["average_generation"],
            "average_predicted_accuracy_cifar100": arch_dict_100[arch_index]["average_predicted_accuracy"],
            "average_generation_cifar100": arch_dict_100[arch_index]["average_generation"],
            "showed_up_cifar10": arch_dict_10[arch_index]["showed_up"],
            "showed_up_cifar100": arch_dict_100[arch_index]["showed_up"]
        })
df_combined = pd.DataFrame(data_combined)

#save to csv
print("Saving to CSV... at /storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas/collecteddata.csv")
#print the first 5 rows of the dataframe
print(df_combined.head())   
df_combined.to_csv('/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas/collecteddata.csv', index=False)
#unify this csv with the one at /storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/nasbench201_corpus_onnx_paper_embedded,csv and save it as collecteddata_unified.csv and also keep all data in the psomu file, and for any
#rows in psomu file that are not in the combined file, fill the predicted accuracy and generation discovery fields with NaN, and the discovery file with 600.
df_psomu = pd.read_csv('/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/nasbench201_corpus_onnx_paper_embedded.csv')
df_unified = pd.merge(df_psomu, df_combined, on='arch_index', how='left')
#fill NaN values in the predicted accuracy and generation discovery fields with 0 and 600 respectively
df_unified['predicted_accuracy_cifar10'] = df_unified['predicted_accuracy_cifar10'].fillna(0)
df_unified['generation_firstdiscovery_cifar10'] = df_unified['generation_firstdiscovery_cifar10'].fillna(600)
df_unified['generation_lastdiscovery_cifar10'] = df_unified['generation_lastdiscovery_cifar10'].fillna(600)
df_unified['predicted_accuracy_cifar100'] = df_unified['predicted_accuracy_cifar100'].fillna(0)
df_unified['generation_firstdiscovery_cifar100'] = df_unified['generation_firstdiscovery_cifar100'].fillna(600)
df_unified['generation_lastdiscovery_cifar100'] = df_unified['generation_lastdiscovery_cifar100'].fillna(600) 
df_unified['average_predicted_accuracy_cifar10'] = df_unified['average_predicted_accuracy_cifar10'].fillna(0)
df_unified['average_generation_cifar10'] = df_unified['average_generation_cifar10'].fillna(600)
df_unified['average_predicted_accuracy_cifar100'] = df_unified['average_predicted_accuracy_cifar100'].fillna(0)
df_unified['average_generation_cifar100'] = df_unified['average_generation_cifar100'].fillna(600)
df_unified['showed_up_cifar10'] = df_unified['showed_up_cifar10'].fillna(0)
df_unified['showed_up_cifar100'] = df_unified['showed_up_cifar100'].fillna(0)
print(df_unified.head())

df_unified.to_csv('/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas/collecteddata_unified.csv', index=False)