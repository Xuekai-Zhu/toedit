import os
import gzip
import json
from tqdm import tqdm
import argparse

def progressive_n(input_dir):
    all_sampled_lines = []
    file_count = 0
    
    # Traverse the directory to get all sampled files
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".json", ".jsonl")):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    
    all_tokens = 0
    
    # Add progress bar for processing files
    for file_path in tqdm(files_to_process, desc="Processing files"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            result = json.loads(lines[0])
            all_tokens += result["tokens"]
    
    print(f"total tokens into {all_tokens}")

def write_to_gz(lines, output_dir, part):
    part_file = os.path.join(output_dir, f"part_{part}.json.gz")
    with gzip.open(part_file, 'wt') as gz_file:
        for line in lines:
            gz_file.write(line)
    print(f"Saved {len(lines)} lines to {part_file}")

if __name__ == "__main__":
    input_dir = "probability/statistics/cot-json-gz-revised-1"
    # output_dir = "data/bio/biomed_lt_0.001_top_p_0.9"
    # chunk_size = 1000000  # Adjust the chunk size as needed
    # os.makedirs(output_dir, exist_ok=True)
    progressive_n(input_dir)
    
    # parser = argparse.ArgumentParser(description="Merge sampled JSONL files into gzipped parts.")
    # parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the sampled JSONL files')
    # parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the merged gzipped files')
    # parser.add_argument('--chunk_size', type=int, default=10000, help='Number of lines per gzipped file')

    # args = parser.parse_args()

    # os.makedirs(args.output_dir, exist_ok=True)
    # merge_sampled_files(args.input_dir, args.output_dir, chunk_size=args.chunk_size)