#!/usr/bin/env python
# eval_vilavt.py - ViLaVT Model Evaluator with Distributed Support

import os
import sys
import json
import pickle
import argparse
import glob


import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from tqdm import tqdm
import subprocess
from datetime import timedelta


# === 工具函数：dump / load（兼容 smp）===
def dump(obj, filepath, **kwargs):
    """Pickle dump with mkdir support"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load(filepath, **kwargs):
    """Pickle load"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# === 全局常量 ===
TYPE_TEMPLATE = {
    "multiple choice": '\nAnswer with the option\'s letter from the given choices directly.',
    "free-form": '',
    "regression": '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).',
    "numerical": '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).',
}

# === 分布式初始化（兼容单进程）===
def setup_distributed():
    if 'RANK' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2)              # ✅ 增加到 2 小时
        )

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_distributed = True
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        is_distributed = False
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, is_distributed


# === 数据集封装 ===
class EvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


# === 构建 prompt 和消息 ===
# def build_message(item, image_folder="", add_type_template=True):
#     prompt = item['question']
#     ptype = item['problem_type'].lower()

#     if ptype == 'multiple choice' and  item.get('options', None) is not None:
#         prompt += '\n' + '\n'.join(item['options'])
#     if add_type_template:
#         prompt += TYPE_TEMPLATE.get(ptype, "")

#     message = []
#     for img_path in item['images']:
#         full_path = os.path.join(image_folder, img_path) if image_folder else img_path
#         message.append({'type': 'image', 'value': full_path})
#     message.append({'type': 'text', 'value': prompt})
#     return message


def build_message(item, image_folder="", add_type_template=True):
    """
    构建消息，支持在prompt中按<image>标签位置插入图片
    
    Args:
        item: 包含question, images, problem_type等字段的字典
        image_folder: 图片文件夹路径
        add_type_template: 是否添加类型模板
        
    Returns:
        message: 包含text和image的消息列表
    """
    prompt = item['question']
    ptype = item['problem_type'].lower()
    
    # 添加选项
    if ptype == 'multiple choice' and item.get('options', None) is not None:
        prompt += '\n' + '\n'.join(item['options'])
    
    # 添加类型模板
    if add_type_template:
        prompt += TYPE_TEMPLATE.get(ptype, "")
    
    message = []
    
    # 检查prompt中是否包含<image>标签
    if '<image>' in prompt:
        # 按<image>标签分割prompt
        parts = prompt.split('<image>')
        
        # 确保图片数量与<image>标签数量匹配
        num_image_tags = prompt.count('<image>')
        num_images = len(item['images'])
        
        if num_image_tags != num_images:
            print(f"Warning: Number of <image> tags ({num_image_tags}) != number of images ({num_images})")
            # 兜底策略：图片数量不匹配时，使用默认方式（所有图片在前）
            for img_path in item['images']:
                full_path = os.path.join(image_folder, img_path) if image_folder else img_path
                message.append({'type': 'image', 'value': full_path})
            message.append({'type': 'text', 'value': prompt.replace('<image>', '')})
            return message
        
        # 交织插入文本和图片
        for i, part in enumerate(parts):
            # 添加文本部分（如果非空）
            if part:
                message.append({'type': 'text', 'value': part})
            
            # 添加图片（最后一个部分后面没有图片）
            if i < len(item['images']):
                full_path = os.path.join(image_folder, img_path) if image_folder else item['images'][i]
                message.append({'type': 'image', 'value': full_path})
    
    else:
        # 无<image>标签：默认所有图片在前面
        for img_path in item['images']:
            full_path = os.path.join(image_folder, img_path) if image_folder else img_path
            message.append({'type': 'image', 'value': full_path})
        message.append({'type': 'text', 'value': prompt})
    
    return message

# === 主要评测函数 ===
def eval_model(args):
    local_rank, rank, world_size, is_distributed = setup_distributed()
    print(f"local_rank, rank, world_size, is_distributed: {local_rank, rank, world_size, is_distributed}")
    # --- 设备与模型 ---
    device = torch.device(f"cuda:{local_rank}")
    # torch.cuda.set_device(device)

    if args.use_baseline:
        from vilavt_baseline import ViLaVT_Baseline as ViLaVT
    else:
        # from vilavt import ViLaVT
        from vilavt_rl import ViLaVT
    
    
    # --- 参数解析 ---
    model_path = args.model_path
    model_name = args.model_name
    input_file = args.input_file
    dataset_name = args.dataset
    output_dir = args.output_dir
    image_folder = args.image_folder if args.image_folder not in ["None", ""] else ""

    # --- 构建路径 ---
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # 最终结果文件
    final_result_file = os.path.join(model_output_dir, f"{model_name}_{dataset_name}_results.jsonl")

    # 如果结果已存在，跳过（支持 reuse）
    if args.reuse and os.path.exists(final_result_file):
        if rank == 0:
            print(f"✅ Result already exists: {final_result_file}, skipping inference.")
        return

    # 临时中间文件：tmp_{rank}_{world_size}_xxx.pkl
    tmp_result_file = os.path.join(
        model_output_dir,
        f"tmp_{rank}_{world_size}_{model_name}_{dataset_name}.pkl"
    )

    # --- 加载数据 ---
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    data=data

    model = ViLaVT(
        model_path=model_path,
        max_iterations=2,
        post_process=True,
        # measure_latency=True,  # ✅ 启用
        verbose=(rank == 0),
        device=device,
        max_pixels=args.max_pixels,
    )
    model.model.eval()

    # --- 数据分片 ---
    dataset = EvalDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=0)

    # --- 推理 ---
    predictions = {}  # idx -> { 'response': ..., 'conversations': ... }

    pbar_desc = f"Rank {rank}/{world_size}"
    total_samples = len(dataloader)

    for (item, original_idx) in tqdm(
        dataloader,
        desc=pbar_desc,
        total=total_samples,
        position=rank if is_distributed else 0,                    # 每个 rank 独立行
        leave=(rank == 0),                     # 完成后清除进度条（整洁）
        ncols=100,                        # 固定宽度
        file=sys.stdout,                  # 必须指定，否则可能不显示
        disable=(rank != 0 and is_distributed)  # 分布式时只有 rank 0 显示？NO！我们想看所有
    ):
        idx = int(original_idx)
        # try:
        add_type_template = not (dataset_name in ["ERQA"])
        message = build_message(item, image_folder, add_type_template)
        response, conversations= model.generate_inner(message, dataset_name)
        idx = int(original_idx)
        predictions[idx] = {
                'response': response,
                'conversations': conversations  # 保存对话历史（如果需要）
        }

    # --- 保存中间结果 ---
    dump(predictions, tmp_result_file)
    if rank == 0:
        print(f"💾 Saved temporary results to {tmp_result_file}")

    # --- 同步与合并 ---
    if world_size > 1:
        print(f"[Rank {rank}] Waiting at barrier...")
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            merged = {}
            pattern = os.path.join(model_output_dir, f"tmp_*_{world_size}_{model_name}_{dataset_name}.pkl")
            for pkl_file in glob.glob(pattern):
                part = load(pkl_file)
                merged.update(part)
                os.remove(pkl_file)

            # 按 index 排序写入最终 JSONL
            with open(final_result_file, 'w', encoding='utf-8') as fout:
                for idx in sorted(merged.keys()):
                    item = data[idx].copy()  # 避免修改原始 data
                    pred = merged[idx]
                    item['response'] = pred['response']
                    item['model_id'] = model_name
                    
                    # 可选：保存中间信息（用于分析）
                    if args.save_intermediate:  # 新增一个命令行参数控制
                        item['conversations'] = pred['conversations']
                    
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"🎉 Final results saved to {final_result_file}")

        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
    else:
        # 单进程模式
        merged = predictions
        with open(final_result_file, 'w', encoding='utf-8') as fout:
            for idx in sorted(merged.keys()):
                item = data[idx]
                item['response'] = merged[idx]['response']
                item['conversations'] = merged[idx]['conversations']
                item['model_id'] = model_name
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        if os.path.exists(tmp_result_file):
            os.remove(tmp_result_file)
        print(f"✅ Debug mode: Results saved to {final_result_file}")


# === 参数解析入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViLaVT model with distributed support.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--use-baseline", action="store_true")
    parser.add_argument("--use-deepeyes", action="store_true")
    parser.add_argument('--save-intermediate', action='store_true', help='Save input message and conversations in final output')
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=8192*28*28)
    args = parser.parse_args()

    eval_model(args)

