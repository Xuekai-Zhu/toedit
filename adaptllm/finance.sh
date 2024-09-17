
MODEL="/home/zhuxuekai/scaling_down_data/continual_training/finance/OLMo-1B-finance/step3240-unsharded-hf" 
HYDRA_FULL_ERROR=1
DOMAIN='finance'

# if the model can fit on a single GPU: set MODEL_PARALLEL=False
# elif the model is too large to fit on a single GPU: set MODEL_PARALLEL=True
MODEL_PARALLEL=False

# number of GPUs, chosen from [1,2,4,8]
N_GPU=1

# # AdaptLLM-7B pre-trained from Llama1-7B
# add_bos_token=False # this is set to False for AdaptLLM, and True for instruction-pretrain
# bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/medicine-LLM' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# # AdaptLLM-13B pre-trained from Llama1-13B
# add_bos_token=False
# bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/medicine-LLM-13B' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# medicine-Llama-8B pre-trained from Llama3-8B in Instruction Pretrain
add_bos_token=False
bash scripts/inference_bigai_4x4090.sh ${DOMAIN} ${MODEL} ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}