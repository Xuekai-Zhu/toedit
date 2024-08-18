import os
import gzip
import json
from tqdm import tqdm

def merge_sampled_files(input_dir, output_dir, chunk_size=10000):
    all_sampled_lines = []
    file_count = 0
    
    # Traverse the directory to get all sampled files
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
    
    # Add progress bar for processing files
    for file_path in tqdm(files_to_process, desc="Processing files"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
                
            all_sampled_lines.extend(lines)
            
            # If the list length exceeds chunk_size, write to a file
            if len(all_sampled_lines) >= chunk_size:
                write_to_gz(all_sampled_lines, output_dir, file_count)
                all_sampled_lines = []
                file_count += 1

    # Write remaining lines if any
    if all_sampled_lines:
        write_to_gz(all_sampled_lines, output_dir, file_count)
    
    print(f"Merged {file_count} files into {output_dir}")

def write_to_gz(lines, output_dir, part):
    part_file = os.path.join(output_dir, f"part_{part}.json.gz")
    with gzip.open(part_file, 'wt') as gz_file:
        for line in lines:
            gz_file.write(line)
    print(f"Saved {len(lines)} lines to {part_file}")

if __name__ == "__main__":
    input_dir = "probability/after_filtering/>0.999"
    output_dir = "data/bio/biomed_delete_greater_than_0.999_token"
    chunk_size = 1000000  # Adjust the chunk size as needed
    os.makedirs(output_dir, exist_ok=True)
    merge_sampled_files(input_dir, output_dir, chunk_size)