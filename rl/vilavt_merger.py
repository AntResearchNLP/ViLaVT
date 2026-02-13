# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForTokenClassification


# usage: python vilavt_merger.py --local_dir checkpoints/rl/vilavt_v0.4.3_ckpt_2100/global_step_200/actor 
# ============ 添加模块路径 ============
module_path = 'path/to/ViLaVT/qwen_vl_finetune/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# ============ 导入自定义模型 ============
try:
    from vilavt import (Qwen2_5_VLForConditionalGeneration_Vilavt, VilavtConfig)
    print("✓ Successfully imported ViLaVT modules")
except ImportError as e:
    print(f"❌ Failed to import ViLaVT modules: {e}")
    print(f"   Module path: {module_path}")
    sys.exit(1)

# ============ 注册自定义模型到 AutoModel ============
AutoConfig.register("vilavt", VilavtConfig)
AutoModelForCausalLM.register(VilavtConfig, Qwen2_5_VLForConditionalGeneration_Vilavt)
AutoModelForVision2Seq.register(VilavtConfig, Qwen2_5_VLForConditionalGeneration_Vilavt)  # 添加这个以支持 Vision2Seq


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()

    assert not args.local_dir.endswith("huggingface"), "The local_dir should not end with huggingface"
    local_dir = args.local_dir

    # ============ 查找分片文件 ============
    print("Scanning for model shards...")
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = int(match.group(1))  # 转换为 int
            break
    
    assert world_size > 0, "No model file with the proper format found"
    print(f"Found world_size: {world_size}")

    # ============ 加载第一个分片获取元信息 ============
    print(f"Loading rank 0 to analyze sharding strategy...")
    state_dict = torch.load(
        os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt"),
        map_location="cpu",
        weights_only=False  # 添加这个参数
    )
    
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor), \
        f"Expected DTensor, got {type(weight)}"
    
    # ============ 获取分片信息 ============
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",),), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing model shards: {total_shards} shards with shape {mesh_shape}")

    # ============ 初始化分片列表 ============
    model_state_dict_lst = [None] * total_shards
    model_state_dict_lst[0] = state_dict

    # ============ 并行加载所有分片 ============
    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️  Warning: Missing file {model_path}")
            return rank, None
        
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return rank, True
        except Exception as e:
            print(f"❌ Failed to load rank {rank}: {e}")
            return rank, None

    print(f"Loading {total_shards} shards in parallel...")
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count()//8)) as executor:
        futures = [executor.submit(process_one_shard, rank) for rank in range(1, total_shards)]
        
        for future in as_completed(futures):
            rank, success = future.result()
            if success:
                print(f"✓ Loaded rank {rank}/{total_shards-1}")
            else:
                print(f"❌ Failed to load rank {rank}")

    # ============ 验证所有分片已加载 ============
    print("Validating all shards...")
    missing_ranks = [i for i, shard in enumerate(model_state_dict_lst) if shard is None]
    if missing_ranks:
        print(f"❌ Missing {len(missing_ranks)} shards: {missing_ranks[:10]}...")
        raise RuntimeError(f"Cannot proceed with missing shards")
    
    print("✓ All shards loaded successfully")

    # ============ 合并分片 ============
    print("Merging shards...")
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    
    for idx, key in enumerate(sorted(keys)):
        if idx % 100 == 0:
            print(f"  Processing parameter {idx}/{len(keys)}: {key}")
        
        state_dict[key] = []
        for shard_idx, model_state_dict in enumerate(model_state_dict_lst):
            if key not in model_state_dict:
                raise KeyError(f"Key '{key}' not found in shard {shard_idx}")
            
            tensor = model_state_dict.pop(key)
            
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == "dp":
                    placements = placements[1:]
                
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements, \
                        f"Inconsistent placements for {key}"
            else:
                state_dict[key] = tensor.bfloat16()
                break  # Non-DTensor, only need first one

    del model_state_dict_lst
    print("✓ Shards merged")

    # ============ 合并参数 ============
    print("Concatenating parameters...")
    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"  Skipping {key} (already merged)")
            continue
        
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print("✓ Parameters concatenated")

    # ============ 保存模型 ============
    print("Writing to local disk...")
    hf_path = os.path.join(local_dir, "huggingface")
    
    # 加载配置
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    print(f"Loaded config: {type(config).__name__}")
    print(f"Architectures: {config.architectures}")

    # ============ 选择正确的模型类 ============
    # if isinstance(config, VilavtConfig) or config.__class__.__name__ == "VilavtConfig":
    #     print("Using Qwen2_5_VLForConditionalGeneration_Vilavt")
    #     auto_model = Qwen2_5_VLForConditionalGeneration_Vilavt
    # elif "ForTokenClassification" in config.architectures[0]:
    #     print("Using AutoModelForTokenClassification")
    #     auto_model = AutoModelForTokenClassification
    # elif "ForCausalLM" in config.architectures[0]:
    #     print("Using AutoModelForCausalLM")
    #     auto_model = AutoModelForCausalLM
    # elif "ForConditionalGeneration" in config.architectures[0]:
    #     print("Using AutoModelForVision2Seq")
    #     auto_model = AutoModelForVision2Seq
    # else:
    #     raise NotImplementedError(f"Unknown architecture {config.architectures}")

    # ============ 创建模型并加载权重 ============
    auto_model = AutoModelForCausalLM
    print("Creating model on meta device...")
    with torch.device("meta"):
        # model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
        model = auto_model.from_config(
            config=config, 
            dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        )

    print("Moving model to CPU...")
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict, safe_serialization=True)
    
    print("✓ Model saved successfully")
    
    del state_dict
    del model

    # ============ 上传到 HuggingFace Hub (可选) ============
    if args.hf_upload_path:
        print(f"Uploading to HuggingFace Hub: {args.hf_upload_path}")
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
            api.upload_folder(folder_path=hf_path, repo_id=args.hf_upload_path, repo_type="model")
            print(f"✓ Model uploaded to {args.hf_upload_path}")
        except Exception as e:
            print(f"❌ Failed to upload to HuggingFace Hub: {e}")
    
    print("\n" + "="*60)
    print("✅ Model merge completed successfully!")
    print(f"   Output: {hf_path}")
    print("="*60)
