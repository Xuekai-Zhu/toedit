import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

# Update these paths to what you want:
data_order_file_path = cached_path("pre_training_results/olma/test/train_data/global_indices.npy")
train_config_path = "OLMo/configs/my_config/test_OLMo.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
print(dataset)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)


def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


# Get all 2048 x 2048 token IDs in the first batch.
i_batch = get_batch_instances(0)
print(i_batch)