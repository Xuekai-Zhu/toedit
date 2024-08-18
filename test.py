from datasets import load_dataset
from datasets import load_from_disk

ds = load_from_disk("data/AI-MO/Edu_filter_Num_CoT")
print(ds)