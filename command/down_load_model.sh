
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Open-Orca/OpenOrca \
  --repo-type dataset \
  --resume-download \
  --local-dir data/Open-Orca/OpenOrca \
  --local-dir-use-symlinks False
