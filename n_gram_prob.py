import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import nltk
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
from nltk.tokenize import word_tokenize
import gzip
from joblib import Parallel, delayed
import pickle

def list_files_in_subdirectories(parent_directory):
    all_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.json.gz'):
                all_files.append(os.path.join(root, file))
    return all_files

def load_text(file_path):
    text_list = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            text_list.append(json.loads(line))
    return text_list

def load_text_in_chunks(file_path, chunk_size=1000):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    # with open(file_path, 'rt') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line)["text"])
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk



def n_grams_prob(in_files):
    local_cfd = ConditionalFreqDist()
    for i, file_i in enumerate(tqdm(in_files, desc="Processing files")):
        for chunk_id, chunk in enumerate(load_text_in_chunks(file_i)):
            for text_i in tqdm(chunk, desc=f"Processing chunk {chunk_id}"):
                tokens = word_tokenize(text_i.lower())
                bigrams = list(nltk.bigrams(tokens))
                
                for bigram in bigrams:
                    local_cfd[bigram[0]][bigram[1]] += 1
                    
    return local_cfd




def merge_cfd(global_cfd, local_cfd):
    for condition in local_cfd.conditions():
        if condition not in global_cfd:
            global_cfd[condition] = local_cfd[condition]
        else:
            for word in local_cfd[condition]:
                global_cfd[condition][word] += local_cfd[condition][word]


def main_step(num_processes, source_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    all_files = list_files_in_subdirectories(source_path)
    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]
    
    results = Parallel(n_jobs=num_processes)(delayed(n_grams_prob)(file_chunk) for file_chunk in file_chunks)
    global_cfd = ConditionalFreqDist()
    for local_cfd in results:
        merge_cfd(global_cfd, local_cfd)
    
    save_object_to_file(global_cfd, os.path.join(output_dir, "cfd.pkl"))

    cpd = ConditionalProbDist(global_cfd, MLEProbDist)
    save_object_to_file(cpd, os.path.join(output_dir, "cpd.pkl"))
    
    
def save_object_to_file(cfd, filename):
    with open(filename, 'wb') as f:
        pickle.dump(cfd, f)

if __name__ == '__main__':
    num_processes = 8
    source_path = "data/math/open-web-math/open-web-math—2B"
    output_dir = f"data/math/open-web-math/open-web-math—2B_n_gram_probs"
    
    main_step(num_processes, source_path, output_dir)
    