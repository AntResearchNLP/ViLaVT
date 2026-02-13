import json
from __init__ import data_dict

# 数据集列表（字符串形式）
data_list = [
    "sr_91k_v2", 
    "spar7m_v2", 
    "vgr_v2", 
    "thyme_2turn_v2", 
    "sr_91k_text", 
    "spar7m_text", 
    "thyme_text_v2", 
    "vica_cot", 
    "vica_text"
]

# 遍历数据集
for dataset in data_list:
    item = data_dict[dataset]
    path = item["annotation_path"]
    
    # 读取并统计
    with open(path, "r") as f:
        data = json.load(f)
        print(f"Dataset: {dataset}, Length: {len(data)}")
