import os
from tqdm import tqdm
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import seaborn as sns
from multiprocessing import Pool, Manager

def list_files_in_subdirectories(parent_directory):
    all_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.jsonl'):
                all_files.append(os.path.join(root, file))
    return all_files

def process_files(files):
    categories = {
        '0.0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
        '0.9-1.0': 0
    }
    total_tokens = 0

    for file in tqdm(files):
        with open(file, "r") as f:
            data = f.readlines()
            for line in data:
                logprobs = json.loads(line)["prompt_logprobs"][1:]
                probs = [math.exp(logprob) for logprob in logprobs]

                for prob in probs:
                    if 0.0 <= prob < 0.1:
                        categories['0.0-0.1'] += 1
                    elif 0.1 <= prob < 0.2:
                        categories['0.1-0.2'] += 1
                    elif 0.2 <= prob < 0.3:
                        categories['0.2-0.3'] += 1
                    elif 0.3 <= prob < 0.4:
                        categories['0.3-0.4'] += 1
                    elif 0.4 <= prob < 0.5:
                        categories['0.4-0.5'] += 1
                    elif 0.5 <= prob < 0.6:
                        categories['0.5-0.6'] += 1
                    elif 0.6 <= prob < 0.7:
                        categories['0.6-0.7'] += 1
                    elif 0.7 <= prob < 0.8:
                        categories['0.7-0.8'] += 1
                    elif 0.8 <= prob < 0.9:
                        categories['0.8-0.9'] += 1
                    elif 0.9 <= prob <= 1.0:
                        categories['0.9-1.0'] += 1

                total_tokens += len(probs)

    return categories, total_tokens

def split_files(files, num_splits):
    return [files[i::num_splits] for i in range(num_splits)]

def process_data_parallel(files, num_processes):
    file_chunks = split_files(files, num_processes)
    with Manager() as manager:
        pool = Pool(processes=num_processes)
        results = pool.map(process_files, file_chunks)
        
        # Combine results
        combined_categories = {
            '0.0-0.1': 0,
            '0.1-0.2': 0,
            '0.2-0.3': 0,
            '0.3-0.4': 0,
            '0.4-0.5': 0,
            '0.5-0.6': 0,
            '0.6-0.7': 0,
            '0.7-0.8': 0,
            '0.8-0.9': 0,
            '0.9-1.0': 0
        }
        total_tokens = 0
        
        for categories, tokens in results:
            for key in combined_categories:
                combined_categories[key] += categories[key]
            total_tokens += tokens

        pool.close()
        pool.join()

    return combined_categories, total_tokens

def plot_pie_chart(categories, total_tokens, save_path):
    labels = list(categories.keys())
    sizes = [count / total_tokens * 100 for count in categories.values()]

    # Use seaborn color palette
    colors = sns.color_palette('coolwarm', len(labels))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=0.85)

    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    plt.title(f'Token Probability Distribution (Total Tokens: {total_tokens})', fontsize=16)

    # Improve legend
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')

    # Add white border between wedges
    for wedge in wedges:
        wedge.set_edgecolor('white')
        wedge.set_linewidth(1)

    out_file = os.path.join(save_path, 'tokens_pie_chart.png')
    plt.savefig(out_file, dpi=500)

def save_statistics(categories, total_tokens, save_path):
    stats = {
        'total_tokens': total_tokens,
        'categories': categories
    }
    stats_file = os.path.join(save_path, 'token_probabilities_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == '__main__':
    parent_directory = 'probability/biomed'
    save_path = "plot/biomed"
    os.makedirs(save_path, exist_ok=True)
    
    files = list_files_in_subdirectories(parent_directory)
    num_processes = 8  
    categories, total_tokens = process_data_parallel(files, num_processes)
    
    save_statistics(categories, total_tokens, save_path)
    plot_pie_chart(categories, total_tokens, save_path)