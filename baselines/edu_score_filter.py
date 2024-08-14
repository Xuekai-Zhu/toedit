from datasets import load_from_disk

# 加载数据集
ds = load_from_disk("baselines/edu_scores")

# 过滤数据集，筛选出 int_score >= 3 的条目
high_score = ds.filter(lambda example: example["int_score"] >= 3)

# 打印前几条记录
for example in high_score.select(range(5)):  # 选择前5条记录
    print(example)

# 保存过滤后的数据集
high_score.save_to_disk("data/AI-MO/Edu_filter_Num_Cot")