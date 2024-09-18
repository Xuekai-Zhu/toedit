
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download HuggingFaceTB/fineweb-edu-classifier \
  --repo-type model \
  --resume-download \
  --local-dir pre_trained_model/HuggingFaceTB/fineweb-edu-classifier \
  --local-dir-use-symlinks False
