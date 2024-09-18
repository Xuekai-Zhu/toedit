import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import math
import json
from tqdm import tqdm
import random
import matplotlib.ticker as ticker

def list_files_in_subdirectories(parent_directory):
    all_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.jsonl'):
                all_files.append(os.path.join(root, file))
    return all_files

def process_data(in_file, sampled=False, sample_num=10):
    all_data = []
    for i in tqdm(in_file):
        with open(i, "r") as f:
            data = f.readlines()
            if sampled:
                data = random.sample(data, sample_num)
            for line in data:
                logprobs = json.loads(line)["prompt_logprobs"][1:]
                probs = [math.exp(logprob) for logprob in logprobs]
                all_data.append(probs)
        
        # dfs = np.array(all_data)
    combined_array = np.concatenate(all_data)
    filtered_array = combined_array[np.isfinite(combined_array)]
    return filtered_array

def plot_ppl_kde(in_values, save_path, upper_percent=99, legend=None, title=None):
    tokens = len(in_values)
    print(f"tital tokens: {tokens}")
    
    # Create the histogram as a density plot
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(in_values, 
             bins=300, 
            #  color='#FF8C00', 
            #  color=color,
             edgecolor='white', 
             linewidth=0.3,
            #  density=True,
             alpha=0.9
             )
    if legend is not None:
        plt.legend([legend], fontsize=15, loc='upper right')
    
    # Optionally add vertical lines for percentiles
    percentiles = [25, 50, 75]
    for pct in percentiles:
        line = np.percentile(in_values, pct)
        plt.axvline(x=line, color="#555555", linestyle='dashed', linewidth=1)
        # plt.text(line, plt.ylim()[1] * 0.8, f'{percentile}th percentile', ha='center', rotation=90)
        plt.text(line, plt.ylim()[1] * 0.5, f'{pct}%', ha='left', va='top', fontsize=15, rotation=90, color="#555555")
    

    # plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add title and labels
    if title:
        plt.title(title, fontsize=23)
    plt.xlabel('Probability', fontsize=23)
    plt.ylabel('Count', fontsize=23)

    # Set tick parameters
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Customize y-axis labels
    # ax = plt.gca()
    # y_labels = ax.get_yticks().tolist()
    # ax.set_yticklabels(['{:.1e}'.format(y) for y in y_labels], fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # Save the figure
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()  # Close the plot to free up memory if running in a loop or on a server


def count_low_prob_elements(array):
    thresholds = [0.01, 0.001, 0.0001]
    low_prob_counts = {threshold: np.sum(array < threshold) for threshold in thresholds}

    print("Counts of elements below thresholds:")
    for threshold, count in low_prob_counts.items():
        print(f"Threshold < {threshold}: {count} elements")

if __name__ == '__main__':
    # Qwen 7B Instruct
    source_path = "/home/xkzhu/scaling_down_data/probability/biomed"
    all_files = list_files_in_subdirectories(source_path)

    all_ppl_vaule = process_data(all_files, sampled=True)
    count_low_prob_elements(all_ppl_vaule)
    
    save_file = "plot/biomed-Qwen2-0.5B.png"
    plot_ppl_kde(all_ppl_vaule, save_file, title="biomed-Qwen2-0.5B")
    
    
