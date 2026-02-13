import json
import os

# 配置
directory = "path/to/ViLaVT/outputs_vllm/"
model_name="ViLaVT"
# dataset = "vsibench_32frame"
# dataset = "viewspatial_bench_2"
# dataset="EmbSpatial_2"
# dataset="spar_bench_2"
# dataset="SpatialEval_spatialreal_2"
# dataset="mmsi_bench"
dataset="ERQA"
num_splits = 1

# 合并所有分片
all_data = []
for i in range(num_splits):
    split_file = os.path.join(
        directory, model_name, 
        f"{model_name}_{dataset}_{i}_{num_splits}",
        f"{model_name}_{dataset}_results.jsonl"
    )
    
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            all_data.extend([json.loads(line) for line in f])
        print(f"✓ Loaded split {i}: {split_file}")
    else:
        print(f"✗ Missing split {i}: {split_file}")

# 保存合并结果
output_path = os.path.join(directory, model_name, f"{model_name}_{dataset}_results.jsonl")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✓ Merged {len(all_data)} items to: {output_path}")
