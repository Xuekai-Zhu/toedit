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

def list_files_in_subdirectories(parent_directory):
    all_files = []
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.jsonl'):
                all_files.append(os.path.join(root, file))
    return all_files

def load_text(file_path):
    with open(file_path, 'rt') as f:
        data = f.readlines()
        text_list = [json.loads(i) for i in data]
    return text_list

def load_text_in_chunks(file_path, chunk_size=1000):
    with open(file_path, 'rt') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def count_tokens(in_files):
    # for i, file_i in enumerate(tqdm(in_files)):
    all_tokens = 0
    for i, file_i in enumerate(tqdm(in_files, desc="Processing files")):
        for chunk_id, chunk in enumerate(load_text_in_chunks(file_i)):
            for entry in chunk:
                token_ids = entry["prompt_token_ids"]
                all_tokens += len(token_ids)

    print(f">>>> all tokens {all_tokens}")
    
    return all_tokens

def main_step1(source_path, num_processes=8):
    """Distribute the token counting process across multiple processes."""
    all_files = list_files_in_subdirectories(source_path)
    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        total_tokens_list = pool.map(count_tokens, file_chunks)

    # Sum tokens from all processes
    total_tokens = sum(total_tokens_list)
    print(f">>>> Total tokens: {total_tokens}")
    
# def main_step1(source_path):
#     num_processes = 8
#     all_files = list_files_in_subdirectories(source_path)

#     file_chunks = [all_files[i::num_processes] for i in range(num_processes)]

#     for i, file_chunk in enumerate(file_chunks):
#         # if strategy == "filter":
#             # strategy_1_filter(file_chunk, tokenizer, output_dir, threshold, i)
#         process = multiprocessing.Process(target=count_tokens, args=([file_chunk]))
#         process.start()

if __name__ == '__main__':
    num_processes = 8
    source_path = "probability/openwebmath"
    main_step1(source_path)