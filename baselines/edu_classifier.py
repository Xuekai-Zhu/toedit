# -------------------- Standard Libraries -------------------- #
import os
import json
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import functools
import csv
# -------------------- Third-Party Libraries -------------------- #
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import wandb
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset


# ------------------------------------- Arguments ------------------------------------- #
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script Configuration")

    # Model Arguments
    parser.add_argument("--model_name_or_path", default="facebook/opt-125m", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str,
                        help="Path to pretrained tokenizer")
    # Add local_rank argument for DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # Data Arguments
    parser.add_argument("--test_file", default=None, type=str,
                        help="An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file).")
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int,
                    help="Batch size for processing the test data.")
    parser.add_argument("--preprocessing_num_workers", default=None, type=int,
                        help="The number of processes to use for the preprocessing.")
    parser.add_argument("--max_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--dataset_name", default="default_dataset", type=str,
                    help="The name of the dataset to be used.")
    parser.add_argument("--num_shards", default=0, type=int, help="Total number of shards in the dataset")
    parser.add_argument("--shard_index", default=0, type=int, help="Index of the current shard being processed")
    parser.add_argument("--pre_tokenized", action='store_true', help="Whether input data is tokenized")

    # Results Arguments
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()
    return args

# ------------------------------------- Utility Functions ------------------------------------- #

def tokenizing_function(args, text_input=None, tokenizer=None):
    padding = "max_length"
    model_inputs = tokenizer(text_input, 
                            max_length=args.max_length, 
                            padding=padding,
                            truncation=True, 
                            return_tensors="pt")
    # model_inputs["labels"] = torch.where(model_inputs["input_ids"] == tokenizer.pad_token_id, -100, model_inputs["input_ids"])
    
    return model_inputs

def search_datasets(args):
    all_train_files = []
    for filename in os.listdir(args.test_file):
        if filename.endswith('.json'):
            file_path = os.path.join(args.test_file, filename)
            all_train_files.append(file_path)
    return all_train_files


class Eval_Dataset(Dataset):
    def __init__(self, json_lines):
        self.prompts = json_lines
    
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.prompts)
    def __getitem__(self, index):
        prompt = self.prompts[index]
        text = prompt["text"]
        text_index = int(prompt["index"])
        return text, text_index

def load_text(file_path):
    with open(file_path, 'rt') as f:
        data = f.readlines()
        text_list = [json.loads(i) for i in data]
    
    return text_list


def move_to_device(data, device):
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


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

def save_ppl_to_csv(results_file, ppl_list, index_list):
    with open(results_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row in the CSV file
        csvwriter.writerow(['PPL', "index"])
        for ppl, index in zip(ppl_list, index_list):
            # if ppl is not None:
            csvwriter.writerow([ppl, index])

def compute_ppl(model_output, labels):
    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(model_output.logits.device)
        # Shift so that tokens < n predict n
        shift_logits = model_output.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Create the loss function
        loss_fct = CrossEntropyLoss(reduction='none')
        
        # Compute the loss per token
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        
        # Average the loss over the batch and sequence dimensions
        loss = loss.mean(dim=-1)
        ppl = torch.exp(loss)

    return ppl

def save_score_to_csv(results_file, score_list, index_list):
    with open(results_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row in the CSV file
        csvwriter.writerow(['score', "index"])
        for score, index in zip(score_list, index_list):
            # if ppl is not None:
            csvwriter.writerow([score, index])

# ------------------------------------- Main Function ------------------------------------- #

def main():
    # Argument Parsing
    args = parse_args()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = torch.device("cuda", args.local_rank)
    # dist.init_process_group(backend="nccl",)
    if args.num_shards != 0:
        output_dir = os.path.join(args.output_dir, f"shard_{args.shard_index}_of_{args.num_shards}")
    else:
        output_dir = args.output_dir
    # Output dir
    if args.local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Model Initialization
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, trust_remote_code=True,)
                                                #  local_files_only=True)
    ds_engine = deepspeed.init_inference(model,
                                 tensor_parallel={"tp_size": world_size},
                                 dtype=torch.half,
                                #  checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                #  replace_with_kernel_inject=True
                                 )
    model = ds_engine.module

    # Tokenizer Initialization
    if args.tokenizer_name_or_path is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, local_files_only=True, add_bos_token=True)
    else:        
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = tokenizer.pad_token_id

    # Test data
    all_train_files = search_datasets(args)
    if args.num_shards != 0:
        all_train_files = shard_list(all_train_files, args.num_shards, args.shard_index)
        print(f"Content of shard {args.shard_index}/{args.num_shards} : {all_train_files}")
    
    # Test Loop
    model.eval()  
    with torch.no_grad(): 
        
        files_to_process = tqdm(all_train_files, desc="Processing files", position=0, leave=False) if args.local_rank == 0 else all_train_files
        for file_path in files_to_process:
            all_scores = []
            all_text_index = []

            all_prompts = load_dataset(file_path)
            # test_dataset = Eval_Dataset(all_prompts)
            # sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=args.local_rank, shuffle=False, drop_last=True)
            # test_dataloader = DataLoader(
            #                 test_dataset,
            #                 batch_size=args.per_device_eval_batch_size, 
            #                 sampler=sampler,
            #                 )
            
            for model_inputs, text_index in tqdm(test_dataloader, desc=f"Test (Rank {args.local_rank})"): 
                
                model_inputs = tokenizing_function(args, text_input=model_inputs, tokenizer=tokenizer).to(device)      
                outputs = model(**model_inputs)
                score = outputs.logits.squeeze(-1).cpu().numpy()
                # score = logits.item()
                all_scores.extend(score.tolist())
                all_text_index.extend(text_index.cpu().numpy().tolist())
                
            # all_scores_np = np.array(all_scores)
            # all_text_index_np = np.array(all_text_index)
            
            base_filename = os.path.basename(file_path)
            results_file = os.path.join(output_dir, f"edu_scores_{base_filename}_rank_{args.local_rank}.csv")
            
            save_score_to_csv(results_file, all_scores, all_text_index)
            
if __name__ == "__main__":
    main()