import os
import json
import csv
from tqdm import tqdm
from vllm import LLM, SamplingParams, RequestOutput
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
import argparse
import gzip
import random 
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import time

# ------------------------------------- Arguments ------------------------------------- #
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script Configuration")

    # Model Arguments
    parser.add_argument("--model_name_or_path", default="facebook/opt-125m", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="For distributed inference")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors.")

    # Data Arguments
    parser.add_argument("--test_file", default=None, type=str,
                        help="An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file).")
    parser.add_argument("--max_length", default=1024, type=int,
                        help="The maximum total output sequence length after tokenization.")
    parser.add_argument("--min_length", default=128, type=int,
                        help="The min total output sequence length after tokenization.")
    # parser.add_argument("--dataset_name", default="default_dataset", type=str,
    #                 help="The name of the dataset to be used.")
    
    parser.add_argument("--num_shards", default=0, type=int, help="Total number of shards in the dataset")
    parser.add_argument("--shard_index", default=0, type=int, help="Index of the current shard being processed")
    parser.add_argument("--batch_size", default=10000, type=int, help="Number of samples to process at a time")
    parser.add_argument("--n_of_candicant", default=8, type=int, help="Number of max_logprobs")
    parser.add_argument("--file_part", choices=['first', 'second'], default=None, required=False,
                        help="Specify which part of the file to process: 'first' or 'second'. If not specified, process the entire file.")
    
    # Results Arguments
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--strategy", default=None, type=str, required=True,
                    help="the strategy to resample or filter out the tokens")

    args = parser.parse_args()
    
    return args

# ------------------------------------- Utility Functions ------------------------------------- #

def calculate_and_store_avg_ppl(outputs):
    all_ppl = []
    for output in outputs:
        probs = output.prompt_logprobs
        prompt_token_ids = output.prompt_token_ids
        
        if probs is None or prompt_token_ids is None:
            all_ppl.append(None)
            # Skip this output if there's no logprob data
            continue
        
        logprob_list = []
        for sample, token_id in zip(probs[1:], prompt_token_ids[1:]):  # Skipping the first None
            first_token_info = sample[token_id]
            logprob_list.append(first_token_info.logprob)

        # if logprob_list:  # Check if list is not empty
        avg_ppl = np.exp(-np.mean(logprob_list))
        all_ppl.append(avg_ppl)

    return all_ppl

def process_vllm_output(output_info):
    logprob_list = []
    # for output in output_info:
    probs = output_info.prompt_logprobs
    prompt_token_ids = output_info.prompt_token_ids
    
    if probs is None or prompt_token_ids is None:
        # Skip this output if there's no logprob data
        return logprob_list
    
    logprob_list.append(None)
    for sample, token_id in zip(probs[1:], prompt_token_ids[1:]):  # Skipping the first None
        first_token_info = sample[token_id]
        logprob_list.append(first_token_info.logprob)

    return logprob_list
    

def save_outputs_to_json(outputs, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for output in outputs:
            print(output)
            exit(0)
            
            logprob_list = process_vllm_output(output)
            output_info = {
                # "text": output.prompt,
                # "generated_text": output.outputs[0].text,
                "prompt_logprobs": logprob_list,
                "prompt_token_ids": output.prompt_token_ids,
            }

            generated_text = json.dumps(output_info)
            f.write(generated_text + "\n")

def save_request_output_as_json(request_output, file_path):

    def convert_logprob_dict(prompt_logprobs):
        logprobs_list = []
        for i in prompt_logprobs:
            if i == None:
                logprobs_list.append(i)
            else:
                converted_dict = {}
                for key, logprob_obj in i.items():
                    converted_dict[key] = {
                        "logprob": logprob_obj.logprob,
                        "rank": logprob_obj.rank,
                        # "decoded_token": logprob_obj.decoded_token
                    }
                logprobs_list.append(converted_dict)
                
        return logprobs_list

    with open(file_path, 'w') as json_file:
        for output in request_output:
            output_dict = {
                "prompt_token_ids": output.prompt_token_ids,
                "prompt_logprobs": convert_logprob_dict(output.prompt_logprobs),
            }

        
            generated_text = json.dumps(output_dict)
            json_file.write(generated_text + "\n")


def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    # e_x = np.exp(x) 
    return e_x / e_x.sum(axis=0)

def online_resampling_json_low_drop_up_revise(request_output, file_path, tokenizer):
    up_threshold = 0.99
    low_threshold = 0.001
    # seconda_threshold = 0.000001
    # seconda_threshold = 0.000001
    
    def resampling(prob_dict, num_samples=1, beta=1):
        words_candicate = list(prob_dict.keys())
        prob_scores = np.array(list(prob_dict.values()))

        adjusted_probs = softmax(prob_scores / beta) 
        accepted_token = np.random.choice(words_candicate, num_samples, p=adjusted_probs)
    
        return accepted_token


    def convert_prob_dict(prompt_logprob):
        converted_dict = {}
        for key, logprob_obj in prompt_logprob.items():
            converted_dict[key] = logprob_obj.logprob
            
        return converted_dict
    
    outputs_to_save = []

    for output in request_output:
        final_tokens = []
        for i_index, prompt_logprob in enumerate(output.prompt_logprobs):

            if prompt_logprob is None:
                final_tokens.append(output.prompt_token_ids[i_index])
            else:
                log_prob_dict = convert_prob_dict(prompt_logprob)
                prob = np.exp(log_prob_dict[output.prompt_token_ids[i_index]])
                
                if prob > up_threshold:
                    accepted_token = resampling(log_prob_dict)
                    final_tokens.append(accepted_token[0])
                elif prob < low_threshold:
                    continue
                else:
                    final_tokens.append(output.prompt_token_ids[i_index])

        revised_text = tokenizer.decode(final_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # source_text = tokenizer.decode(output.prompt_token_ids)
        output_dict = {"text": revised_text,} #"source_text":source_text}
        outputs_to_save.append(output_dict) 
            
    with open(file_path, 'w') as json_file:
        for output_res in outputs_to_save:
            generated_text = json.dumps(output_res, ensure_ascii=False)
            json_file.write(generated_text + "\n")


def online_resampling_json_up_revise(request_output, file_path, tokenizer, source_in_texts):
    up_threshold = 0.99
    low_threshold = 0.001
    # seconda_threshold = 0.000001
    # seconda_threshold = 0.000001
    
    def resampling(prob_dict, num_samples=1, beta=1.5):
        words_candicate = list(prob_dict.keys())
        prob_scores = np.array(list(prob_dict.values()))

        adjusted_probs = softmax(prob_scores / beta) 
        accepted_token = np.random.choice(words_candicate, num_samples, p=adjusted_probs)
    
        return accepted_token


    def convert_prob_dict(prompt_logprob):
        converted_dict = {}
        for key, logprob_obj in prompt_logprob.items():
            converted_dict[key] = logprob_obj.logprob
            
        return converted_dict
    
    outputs_to_save = []

    for i, output in enumerate(request_output):
        final_tokens = []
        for i_index, prompt_logprob in enumerate(output.prompt_logprobs):

            if prompt_logprob is None:
                final_tokens.append(output.prompt_token_ids[i_index])
            else:
                log_prob_dict = convert_prob_dict(prompt_logprob)
                prob = np.exp(log_prob_dict[output.prompt_token_ids[i_index]])
                
                if prob > up_threshold:
                    accepted_token = resampling(log_prob_dict)
                    final_tokens.append(accepted_token[0])
                # elif prob < low_threshold:
                #     continue
                else:
                    final_tokens.append(output.prompt_token_ids[i_index])

        revised_text = tokenizer.decode(final_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # source_text = tokenizer.decode(output.prompt_token_ids)
        output_dict = {"input":source_in_texts[i], "output": revised_text,} #"source_text":source_text}
        outputs_to_save.append(output_dict) 
            
    with open(file_path, 'w') as json_file:
        for output_res in outputs_to_save:
            generated_text = json.dumps(output_res, ensure_ascii=False)
            json_file.write(generated_text + "\n")


def filtering_out(request_output, file_path, tokenizer):
    threshold = 0.001
    
    def convert_prob_dict(prompt_logprob):
        converted_dict = {}
        for key, logprob_obj in prompt_logprob.items():
            converted_dict[key] = logprob_obj.logprob
        return converted_dict
    
    outputs_to_save = []  # List to store all output dictionaries

    for output in request_output:
        final_tokens = []
        for i_index, prompt_logprob in enumerate(output.prompt_logprobs):

            if prompt_logprob is None:
                final_tokens.append(output.prompt_token_ids[i_index])
            else:
                log_prob_dict = convert_prob_dict(prompt_logprob)
                prob = np.exp(log_prob_dict[output.prompt_token_ids[i_index]])
                
                if prob >= threshold:
                    final_tokens.append(output.prompt_token_ids[i_index])

        revised_text = tokenizer.decode(final_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_dict = {"text": revised_text}
        outputs_to_save.append(output_dict)  # Collect the output dict for later saving
    
    # Write all outputs to file
    with open(file_path, 'w') as json_file:
        for output_res in outputs_to_save:
            generated_text = json.dumps(output_res, ensure_ascii=False)
            json_file.write(generated_text + "\n")

    
    

def truncate_texts(tokenizer, texts, max_length=None):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if isinstance(texts, str):
        texts = [texts]

    truncated_texts = []

    for text in texts:
        inputs = tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
        )

        truncated_text = tokenizer.decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        truncated_texts.append(truncated_text)

    return truncated_texts[0] if isinstance(texts, str) else truncated_texts

    
def save_ppl_to_csv(results_file, ppl_list):
    with open(results_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row in the CSV file
        csvwriter.writerow(['PPL'])
        for ppl in ppl_list:
            # if ppl is not None:
            csvwriter.writerow([ppl if ppl is not None else 'N/A'])
              
      

def search_datasets(args):
    all_train_files = []
    for filename in os.listdir(args.test_file):
        if filename.endswith('.gz'):
            file_path = os.path.join(args.test_file, filename)
            all_train_files.append(file_path)
    return all_train_files

def read_gz_file(path):
    """
    Read a gzipped JSON file line by line, yielding each JSON object.
    """
    with gzip.open(path, 'rt') as f:
        # for line in f:
        data = f.readlines()
    texts = []
    for line in data:
        json_line = json.loads(line)
        texts.append(json_line["text"])
    return texts


def read_gz_sft_file(path):
    """
    Read a gzipped JSON file line by line, yielding each JSON object.
    """
    with gzip.open(path, 'rt') as f:
        # for line in f:
        data = f.readlines()
    in_texts = []
    out_texts = []
    for line in data:
        json_line = json.loads(line)
        in_texts.append(json_line["input"])
        out_texts.append(json_line["output"])
    return in_texts, out_texts

class Pormpt_Dataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        prompt = self.prompts[index]
        return prompt
 
 
def shard_list(data_list, num_shards, shard_index):
    if num_shards <= 0 or shard_index < 1 or shard_index > num_shards:
        raise ValueError("Invalid number of shards or shard index.")

    sorted_list = sorted(data_list)
    shard_size = len(sorted_list) // num_shards
    remainder = len(sorted_list) % num_shards

    start_index = 0
    shards = []

    for i in range(num_shards):
        end_index = start_index + shard_size + (1 if i < remainder else 0)
        shards.append(sorted_list[start_index:end_index])
        start_index = end_index
        
    return shards[shard_index-1]
      
# ------------------------------------- Main Function ------------------------------------- #

def main():
    # Argument Parsing
    args = parse_args()

    # Output dir
    # results_dir = os.path.join(args.model_name_or_path, "ppl")
    if args.num_shards != 0:
        output_dir = os.path.join(args.output_dir, f"shard_{args.shard_index}_of_{args.num_shards}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # file_count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
    
    
    # load data
    all_train_files = search_datasets(args)
    if args.num_shards != 0 and len(all_train_files) > 1:
        all_train_files = shard_list(all_train_files, args.num_shards, args.shard_index)
        print(f"Content of shard {args.shard_index}/{args.num_shards} : {all_train_files}")
        
    files_to_process = tqdm(all_train_files, desc="Processing files", position=0, leave=False)
    # Model Initialization
    llm = LLM(model=args.model_name_or_path, 
              tensor_parallel_size=args.tensor_parallel_size, 
              gpu_memory_utilization=args.gpu_memory_utilization,  
              max_model_len=2048,
              max_logprobs=args.n_of_candicant
            #   skip_tokenizer_init=True,
              )
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(max_tokens=1, 
                                    #  min_tokens=args.min_length
                                    prompt_logprobs=args.n_of_candicant,
                                    #  prompt_logprobs=1,
                                    #  detokenize=False, 
                                     # n=1,  temperature=0.7, top_p=0.8, repetition_penalty=1.05, top_k=20, 
                                    )
    global_files = 0
    for i, file_path in enumerate(files_to_process):
        # all_prompts = read_gz_file(file_path)
        in_texts, out_texts = read_gz_sft_file(file_path)
        
        # if args.file_part:
        #     half_index = len(all_prompts) // 2
        #     if args.file_part == "first":
        #         all_prompts = all_prompts[:half_index]
        #     elif args.file_part == "second":
        #         all_prompts = all_prompts[half_index:]
        #     print(f"Processing {args.file_part} half of the prompts")
        
        # if len(all_train_files) == 1 and args.num_shards != 0 and "math" in args.test_file: 
        #     all_prompts = all_prompts[args.shard_index-1::args.num_shards]
        #     print(f">>>>>> Shard {args.shard_index} of {args.num_shards} in one file")
            
        
        for start in tqdm(range(0, len(out_texts), args.batch_size), desc=f"Processing batch"):
            
            # restart from check points
            global_files += 1
            # if global_files <= file_count:
            #     continue
            batch_intexts = in_texts[start:start+args.batch_size]
            batch_prompts = out_texts[start:start+args.batch_size]
            prompt_token_ids = tokenizer(batch_prompts, truncation=True, max_length=2048)["input_ids"]
        
            try:
                outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
            except Exception as e:
                print(f"Error generating outputs for batch {start//args.batch_size} in file {file_path}: {e}")
                continue
        
            model_name = args.model_name_or_path.split("/")[-1]
            timestamp = int(time.time() * 1000) 
            results_file = os.path.join(output_dir, f"{i}_{model_name}_{start//args.batch_size}_{args.file_part}_{timestamp}.jsonl")
            # save_request_output_as_json(outputs, results_file)
            # save_outputs_to_json(outputs, results_file)
            
            if args.strategy == "low_drop_up_revise":
                online_resampling_json_low_drop_up_revise(outputs, results_file, tokenizer)
            elif args.strategy == "up_revise":
                online_resampling_json_up_revise(outputs, results_file, tokenizer, batch_intexts)
            elif args.strategy == "filtering":
                filtering_out(outputs, results_file, tokenizer)
            else:
                print(f"Error: Unsupported strategy '{args.strategy}' specified. Please choose either 'revised' or 'filtering'.")

            
if __name__ == "__main__":
    main()