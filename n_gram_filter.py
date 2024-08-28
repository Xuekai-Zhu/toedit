import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import gzip
import multiprocessing
import torch
import random
import argparse
import pickle
from nltk import word_tokenize
import heapq


def list_files_in_subdirectories(parent_directory):
    all_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.gz'):
                all_files.append(os.path.join(root, file))
    return all_files

def load_text(file_path):
    with open(file_path, 'rt') as f:
        data = f.readlines()
        text_list = [json.loads(i) for i in data]
    return text_list

def load_text_in_chunks(file_path, chunk_size=1000):
    with gzip.open(file_path, 'rt') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def load_object_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0)

def reject_sampling(possible_words_sorted, prob_dist, candicate_num=64, num_samples=1, beta=0.5):
    
    candidates = {w: prob_dist.prob(w) for w in possible_words_sorted[:candicate_num]}
    
    rewards = np.array(list(candidates.values()))
    adjusted_probs = softmax(rewards / beta) 
    
    accepted = []
    while len(accepted) < num_samples:
        to_remove = []
        
        for idx, (candidate, reward) in enumerate(candidates.items()):
            u = np.random.uniform()  
            if u >= adjusted_probs[idx]:  
                continue
    
            accepted.append(candidate) 
            to_remove.append(candidate)  

            if len(accepted) == num_samples:  
                break
        for c in to_remove:
            candidates.pop(c)

    return accepted


def random_chioce(possible_words_sorted, prob_dist, candicate_num=64, num_samples=1, beta=0.5):
    candidates = {w: prob_dist.prob(w) for w in possible_words_sorted[:candicate_num]}
    
    rewards = np.array(list(candidates.values()))
    adjusted_probs = softmax(rewards / beta) 
    words_candicate = np.array(list(candidates.keys()))
    
    accepted_token = np.random.choice(words_candicate, num_samples, p=adjusted_probs)
    
    return accepted_token
    
def get_top_n_samples(prob_dist, top_n):
    samples = list(prob_dist.samples())
    probs = [prob_dist.prob(w) for w in samples]
    
    top_n_indices = heapq.nlargest(top_n, range(len(probs)), probs.__getitem__)
    top_n_samples = [(samples[i], probs[i]) for i in top_n_indices]
    
    return top_n_samples


def main_step1(num_processes, source_path, output_dir, cpd_file_path=None, threshold=None):
    os.makedirs(output_dir, exist_ok=True)
    all_files = list_files_in_subdirectories(source_path)
    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]
    
    n_gram_cpd = load_object_from_file(cpd_file_path)

    for i, file_chunk in enumerate(file_chunks):
        # strategy_1_filter(file_chunk, output_dir, n_gram_cpd, threshold, i)
        process = multiprocessing.Process(target=strategy_1_filter, args=(file_chunk, output_dir, n_gram_cpd, threshold, i))
        process.start()
        
def strategy_1_filter(in_files, out_dir, n_gram_cpd, threshold, process_id):
    for i, file_i in enumerate(tqdm(in_files, desc="Processing files")):
        # processed files into chunks
        for chunk_id, chunk in enumerate(load_text_in_chunks(file_i)):
            results = []
            revised_num = []
            # process each sample
            for entry in tqdm(chunk, desc=f"Processing entries in chunk {chunk_id}"):
                text = entry["text"]
                tokens = word_tokenize(text.lower())  
                # token_probs = {}
                final_tokens = [tokens[0]]
                n = 0
                
                # filter tokens
                for j in range(1, len(tokens)):
                    context = tokens[j-1]
                    word = tokens[j]
                    prob = n_gram_cpd[context].prob(word) #if context in cpd else 0.0
                    
                    if prob < threshold:
                        prob_dist = n_gram_cpd[context]
                        top_n_samples = get_top_n_samples(prob_dist, 64)
                        possible_words_sorted = [w for w, _ in top_n_samples]
                        # _possible_words = list(prob_dist.samples())
                        # _possible_words_sorted = sorted(_possible_words, key=lambda w: prob_dist.prob(w), reverse=True)
                        # accept_token = reject_sampling(possible_words_sorted, prob_dist)
                        accept_token = random_chioce(possible_words_sorted, prob_dist)
                        
                        final_tokens.append(accept_token[0])
                        n+=1
                    else:
                        final_tokens.append(word)
                        
                results.append({"text": " ".join(final_tokens)})
                revised_num.append(n)
                
            avg_token = np.mean(revised_num)
            print(f">>>>>>>>>>>>> {len(results)} smaples revised {avg_token} tokens" + "\n\n\n" )
                
            random_number = random.randint(1000, 9999)
            out_file = os.path.join(out_dir, f"n_gram_process_{process_id}_file_{i}_chunk_{chunk_id}_filtered_token_lt_{threshold}_{random_number}.jsonl")
            with open(out_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
    
if __name__ == '__main__':
    num_processes = 11
    
    # bio
    threshold = 0.000001
    source_path = "data/bio/instruction_biomed"
    output_dir = f"data/bio/instruction_biomed_n_gram_filtering_lt{threshold}"
    cpd_file_path = "data/bio/instruction_biomed_n_gram_probs/cpd.pkl"
    
    main_step1(num_processes, source_path, output_dir, cpd_file_path=cpd_file_path, threshold=threshold)
    
    
    # >>>>>>>>>>>>>>>>>>>>>>>
    # parser = argparse.ArgumentParser(description='Process some files with different strategies.')
    # parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    # parser.add_argument('--source_path', type=str, required=True, help='Path to the source files')
    # parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files')
    # parser.add_argument('--strategy', type=str, choices=['filter', 'top_p'], required=True, help='Strategy to use (filter or top_p)')
    # parser.add_argument('--threshold', type=float, required=True, help='Threshold for filtering')
    # parser.add_argument('--top_p', type=float, default=None, help='Top-p value (only required if strategy is top_p)')
    
    # args = parser.parse_args()
    # main_step1(
    #     num_processes=args.num_processes,
    #     source_path=args.source_path,
    #     output_dir=args.output_dir,
    #     strategy=args.strategy,
    #     threshold=args.threshold,
    #     top_p=args.top_p
    # )
    