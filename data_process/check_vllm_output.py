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

def main_step1(source_path):
    num_processes = 8
    all_files = list_files_in_subdirectories(source_path)

    file_chunks = [all_files[i::num_processes] for i in range(num_processes)]

    for i, file_chunk in enumerate(file_chunks):
        # if strategy == "filter":
            # strategy_1_filter(file_chunk, tokenizer, output_dir, threshold, i)
        process = multiprocessing.Process(target=count_tokens, args=([file_chunk]))
        process.start()


def count_tokens(in_files):
    # for i, file_i in enumerate(tqdm(in_files)):
    all_tokens = 0
    for i, file_i in enumerate(tqdm(in_files, desc="Processing files")):
        for chunk_id, chunk in enumerate(load_text_in_chunks(file_i)):
            for entry in chunk:
                token_ids = entry["prompt_token_ids"]
                all_tokens += len(token_ids)

    print(f">>>> all tokens {all_tokens}")
    


if __name__ == '__main__':
    num_processes = 8
    
    # bio
    # strategy = "filter"
    # threshold = 0.001
    # source_path = "probability/biomed_8"
    # output_dir = f"probability/biomed_8<{threshold}"
    
    # main_step1(num_processes, source_path, output_dir, strategy=strategy, threshold=threshold)
    
    # strategy = "top_p"
    # threshold = 0.001
    # top_p = 0.9
    # source_path = "probability/biomed_8"
    # output_dir = f"probability/biomed_8_filtering/lt_{threshold}_top_p_0.9"
    # # output_dir = f"test/lt_{threshold}_top_p_0.9"
    
    # main_step1(num_processes, source_path, output_dir, strategy=strategy, threshold=threshold, top_p=top_p)
    
    # open web math
    source_path = "probability/openwebmath"
    # files = list_files_in_subdirectories(source_path)

    main_step1(source_path)
    
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
    