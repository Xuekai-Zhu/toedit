import os
import gzip
import json
from tqdm import tqdm
import argparse

def add_instruction(input_dir, output_dir):
    all_sampled_lines = []
    file_count = 0
    
    # Traverse the directory to get all sampled files
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    
    # Add progress bar for processing files
    for file_path in tqdm(files_to_process, desc="Processing files"):
        file_name = os.path.basename(file_path)
        out_file = os.path.join(output_dir, file_name)
        
        with open(out_file, "w", encoding="utf-8") as f_out:
            with open(file_path, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                
                for entry in lines:
                    entry_dict = json.loads(entry)
                    entry_dict["instruction"] = ""
                    item = json.dumps(entry_dict) + "\n"
                    f_out.write(item)

#     # Write remaining lines if any
#     if all_sampled_lines:
#         write_to_gz(all_sampled_lines, output_dir, file_count)
    
#     print(f"Merged {file_count} files into {output_dir}")

# def write_to_gz(lines, output_dir, part):
#     part_file = os.path.join(output_dir, f"part_{part}.json.gz")
#     with gzip.open(part_file, 'wt') as gz_file:
#         for line in lines:
#             gz_file.write(line)
#     print(f"Saved {len(lines)} lines to {part_file}")

if __name__ == "__main__":
    input_dir = "data/less-data-llama-revised-2/cot-json-gz-copy"
    output_dir = "data/less-data-llama-revised-2/cot-json-gz-copy-add-instruction"

    os.makedirs(output_dir, exist_ok=True)
    add_instruction(input_dir, output_dir)
    
