#!/bin/bash
#SBATCH -J olma_tokenize
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

source /home/zhuxuekai/miniconda3/bin/activate py310

python OLMo/scripts/prepare_memmap_dataset.py data/dolma/dolma_sampled/*.json.gz -o data/dolma/olma_tokenize/