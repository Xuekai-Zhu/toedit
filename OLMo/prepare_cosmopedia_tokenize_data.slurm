#!/bin/bash
#SBATCH -J olma_tokenize
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

source /home/zhuxuekai/miniconda3/bin/activate py310

python OLMo/scripts/prepare_memmap_dataset.py data/cosmopedia/cosmopedia_sampled/cosmopedia_10B_json_dir/*.json.gz -o data/cosmopedia/cosmopedia_sampled/10B_tokenize_2048