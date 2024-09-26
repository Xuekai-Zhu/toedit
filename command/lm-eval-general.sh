MODEL="continual_training/dolma/OLMo-1B-dolma/step20000-unsharded-hf"

# python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
#      --input_dir ${MODEL} \
#      --tokenizer_json_path OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
#      --output_dir ${MODEL}-hf

# export HF_ENDPOINT=https://hf-mirror.com
# export HF_DATASETS_TRUST_REMOTE_CODE=1

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks hellaswag,arc_easy,arc_challenge,social_iqa,winogrande,piqa,boolq,openbookqa \
    --batch_size 16 \
    --output_path eval_results/general/ \
    --trust_remote_code