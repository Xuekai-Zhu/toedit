import os
import random

in_path = "data/bio/biomed_revised+orca_tokenized"
# dolma_path = "data/dolma/olma_tokenize"

files = [os.path.join(in_path, i) for i in os.listdir(in_path)]
files = sorted(files)
# dolma_file_path = [os.path.join(dolma_path, i) for i in os.listdir(dolma_path)]


# # 比例设置
# cosmopedia_ratio = 0.75
# dolma_ratio = 1 - cosmopedia_ratio

# # 计算每个列表中选取的元素数量
# num_cosmopedia = int(cosmopedia_ratio * len(cosmopedia_file_path))
# num_dolma = int(dolma_ratio * len(dolma_file_path))

# # 随机选择文件
# selected_cosmopedia = random.sample(cosmopedia_file_path, num_cosmopedia)
# selected_dolma = random.sample(dolma_file_path, num_dolma)

# # 合并两个列表
# mixed_list = selected_cosmopedia + selected_dolma

# # 如果需要，可以随机打乱最终列表的顺序
# random.shuffle(mixed_list)

# 打印或返回新列表
for item in files:
    print("- " + item)