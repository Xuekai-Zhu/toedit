import numpy as np
import os
from tqdm import tqdm

def list_files(directory):
    """
    List all files in a given directory, including files in subdirectories.

    :param directory: The path to the directory to search.
    :return: A list of paths to the files.
    """
    file_paths = []  # List to store file paths
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):  # Only consider .npy files
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

def calculate_total_size_and_tokens(directory):
    """
    Calculate the total size and number of tokens in all .npy files within a directory.

    :param directory: The path to the directory to search.
    :return: A tuple containing the total size in bytes and the total number of tokens.
    """
    total_size = 0
    total_tokens = 0

    file_paths = list_files(directory)

    for file_path in tqdm(file_paths):
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
        total_size += data.nbytes
        total_tokens += data.size

    return total_size, total_tokens

# Path to the directory containing .npy files
directory = 'data/bio/instruction_biomed_tokenized'

# Calculate the total size and number of tokens
total_size, total_tokens = calculate_total_size_and_tokens(directory)

print(f"Total Size: {total_size / (1024 ** 3):.2f} GB")  # Convert bytes to gigabytes
print(f"Total Number of Tokens: {total_tokens}")
