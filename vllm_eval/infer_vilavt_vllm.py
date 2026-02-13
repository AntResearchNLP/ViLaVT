import argparse
import traceback
import random
import re
import copy
import torch
import os
import json
from torch._C import NoneType
from tqdm import tqdm
import pdb
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info, fetch_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import parse_output, zoomin
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from datetime import datetime
import time
import uuid

from vllm.multimodal.inputs import MultiModalKwargs, BatchedTensorInputs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 保存原始的 batch 方法
_original_batch_method = MultiModalKwargs.batch

def tensor_to_uuid(tensor: torch.Tensor) -> str:
    """
    将 tensor 转换回 UUID 字符串
    
    Args:
        tensor: shape=[16], dtype=torch.uint8
        
    Returns:
        UUID 字符串
    """
    uuid_bytes = bytes(tensor.cpu().numpy())
    return str(uuid.UUID(bytes=uuid_bytes))


def set_seed(seed: int = 42) -> None:
    """
    设置多个库的随机种子，以确保实验的可复现性。

    Args:
        seed (int): 随机种子值，默认为42。
    """
    # 1. Python 内置的 random 模块
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    # 如果使用 CUDA (GPU)，也需要设置 CUDA 相关的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 为所有可用的 GPU 设置种子
        
        # 进一步确保 CUDA 操作的确定性（可能略微降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 如果设置为True，可能导致非确定性
    
    # 4. 其他环境设置（某些库可能会从这里获取随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"所有相关的随机种子已设置为: {seed}")



def extract_answer(text):
    if not isinstance(text, str):
        # 如果不是字符串，尝试转换；若无法转换，返回空字符串或 None
        if text is None:
            return ""
        try:
            text = str(text)
        except Exception:
            return ""

    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

# def _patched_batch_method(
#         inputs_list: list[MultiModalKwargs], 
#         pin_memory: bool = False  # <-- 修复：添加与原始方法一致的 pin_memory 参数
#     ) -> BatchedTensorInputs:
#     """
#     一个被修复的 batch 方法。
#     它能识别共享字段，避免对它们进行堆叠，
#     然后将它们与正常批处理后的字段合并。
#     """
#     if len(inputs_list) == 0:
#         return {}
#     # a. 定义你的共享字段的键名
#     shared_keys = {"query_ids", "query_attention_mask", "image_to_query_mapping", "request_uuid"}
#     # b. 提取共享数据 (只从第一个请求中取，因为它们都一样), 并同时准备用于调用原始批处理方法的数据
#     inputs_without_shared = []
    
#     seen_uuid_keys = set()
#     all_query_ids = []
#     all_query_masks = []
#     all_image_mappings = []
#     offset = 0
#     for idx, inputs_kwargs in enumerate(inputs_list):
#         request_uuid_tensor = inputs_kwargs.get("request_uuid", None)
        
#         if request_uuid_tensor is None:
#             # print(f"  ⚠️  输入 {idx} 缺少 request_uuid")
#             uuid_key = f"auto_{idx}"
#         else:
#             # uint8 tensor -> tuple (用作字典 key)
#             uuid_key = tuple(request_uuid_tensor.reshape(-1).tolist())
#             # 转换回 UUID 字符串用于调试
#             try:
#                 uuid_str = tensor_to_uuid(request_uuid_tensor.reshape(-1))
#             except:
#                 uuid_str = str(uuid_key[:4]) + "..."  # 显示前几个字节
#          # 检查是否已处理过这个 UUID
#         if uuid_key in seen_uuid_keys:
#             # print(f"  └─ 输入 {idx}: 共享 query，跳过")
#             continue

#         seen_uuid_keys.add(uuid_key)
#         q_ids = inputs_kwargs["query_ids"]
#         q_mask = inputs_kwargs["query_attention_mask"]
#         img_mapping = inputs_kwargs["image_to_query_mapping"]
#         num_queries = q_ids.shape[0]

#         all_query_ids.append(q_ids)
#         all_query_masks.append(q_mask)
#         adjusted_mapping = img_mapping + offset
#         all_image_mappings.append(adjusted_mapping)
#         offset = offset + num_queries

#     batched_query_ids = torch.cat(all_query_ids, dim=0)  # [total_queries, seq_len]
#     batched_query_masks = torch.cat(all_query_masks, dim=0)
#     batched_image_mapping = torch.cat(all_image_mappings, dim=0)  # [total_images]
#     # batched_query_ids = inputs_list[0]["query_ids"]
#     # batched_query_masks = inputs_list[0]["query_attention_mask"]
#     # batched_image_mapping = inputs_list[0]["image_to_query_mapping"]

#     # 遍历所有输入，创建不含共享字段的新输入列表
#     for inputs in inputs_list:
#         per_item_data = {k: v for k, v in inputs.items() if k not in shared_keys}
#         inputs_without_shared.append(MultiModalKwargs(per_item_data))

#     # c. 对剩下的、需要堆叠的数据调用原始的 batch 方法
#     batched_per_item_data = _original_batch_method(
#         inputs_without_shared, pin_memory=pin_memory # <-- 修复：传递 pin_memory 参数
#     )
#     # d. 将共享数据添加回来，完成最终的批处理字典
#     batched_per_item_data.update({
#         'query_ids': batched_query_ids,
#         'query_attention_mask': batched_query_masks,
#         'image_to_query_mapping': batched_image_mapping,
#     })
    
#     return batched_per_item_data


# def _patched_batch_method(
#         inputs_list: list[MultiModalKwargs], 
#         pin_memory: bool = False  # <-- 修复：添加与原始方法一致的 pin_memory 参数
#     ) -> BatchedTensorInputs:
#     """
#     一个被修复的 batch 方法。
#     它能识别共享字段，避免对它们进行堆叠，
#     然后将它们与正常批处理后的字段合并。
#     """
#     if len(inputs_list) == 0:
#         return {}
#     # a. 定义你的共享字段的键名
#     shared_keys = {"query_ids", "query_attention_mask", "image_to_query_mapping", "request_uuid"}
#     # b. 提取共享数据 (只从第一个请求中取，因为它们都一样), 并同时准备用于调用原始批处理方法的数据
#     inputs_without_shared = []
    
#     seen_uuid_keys = set()
#     all_query_ids = []
#     all_query_masks = []
#     all_image_mappings = []
#     offset = 0
#     print(f"{len(inputs_list)}-inputs_list: {inputs_list}")
#     for idx, inputs_kwargs in enumerate(inputs_list):
#         request_uuid_tensor = inputs_kwargs.get("request_uuid", None)
        
#         if request_uuid_tensor is None:
#             # print(f"  ⚠️  输入 {idx} 缺少 request_uuid")
#             uuid_key = f"auto_{idx}"
#         else:
#             # uint8 tensor -> tuple (用作字典 key)
#             uuid_key = tuple(request_uuid_tensor.reshape(-1).tolist())
#             # 转换回 UUID 字符串用于调试
#             try:
#                 uuid_str = tensor_to_uuid(request_uuid_tensor.reshape(-1))
#             except:
#                 uuid_str = str(uuid_key[:4]) + "..."  # 显示前几个字节
#          # 检查是否已处理过这个 UUID
#         if uuid_key in seen_uuid_keys:
#             # print(f"  └─ 输入 {idx}: 共享 query，跳过")
#             continue

#         seen_uuid_keys.add(uuid_key)
#         q_ids = inputs_kwargs["query_ids"]
#         q_mask = inputs_kwargs["query_attention_mask"]
#         img_mapping = inputs_kwargs["image_to_query_mapping"]
#         num_queries = q_ids.shape[0]

#         all_query_ids.append(q_ids)
#         all_query_masks.append(q_mask)
#         adjusted_mapping = img_mapping + offset
#         all_image_mappings.append(adjusted_mapping)
#         offset = offset + num_queries

#     batched_query_ids = torch.cat(all_query_ids, dim=0)  # [total_queries, seq_len]
#     batched_query_masks = torch.cat(all_query_masks, dim=0)
#     batched_image_mapping = torch.cat(all_image_mappings, dim=0)  # [total_images]
#     # batched_query_ids = inputs_list[0]["query_ids"]
#     # batched_query_masks = inputs_list[0]["query_attention_mask"]
#     # batched_image_mapping = inputs_list[0]["image_to_query_mapping"]

#     # 遍历所有输入，创建不含共享字段的新输入列表
#     for inputs in inputs_list:
#         per_item_data = {k: v for k, v in inputs.items() if k not in shared_keys}
#         inputs_without_shared.append(MultiModalKwargs(per_item_data))

#     # c. 对剩下的、需要堆叠的数据调用原始的 batch 方法
#     batched_per_item_data = _original_batch_method(
#         inputs_without_shared, pin_memory=pin_memory # <-- 修复：传递 pin_memory 参数
#     )
#     # d. 将共享数据添加回来，完成最终的批处理字典
#     batched_per_item_data.update({
#         'query_ids': batched_query_ids,
#         'query_attention_mask': batched_query_masks,
#         'image_to_query_mapping': batched_image_mapping,
#     })
    
#     return batched_per_item_data


# 3. 用我们的新方法替换掉 vLLM 原始的静态方法
# MultiModalKwargs.batch = staticmethod(_patched_batch_method)
# print("✅ Monkey-patch for MultiModalKwargs.batch has been applied.")


REASONING_SYSTEM_PROMPT = """You are a helpful assistant. 
Your goal is to solve the problem in the provided image(s) based on the user's instruction. Proceed step by step, optionally using the zoom-in tool one or more times to examine key areas closely. Selected regions will be cropped and processed externally, then re-encoded with your query to extract critical details.

# Tools
If needed, use the zoom-in tool one or more times to examine specific areas in detail.

## Tool Format
Structure:
{
    "region": [
        {
            "index": int,       # Target image index to zoom in (0-based)
            "bbox_2d": list,    # Format: [x1, y1, x2, y2], where (x1, y1) is top-left corner and (x2, y2) is bottom-right corner
        },
        ...                    # Additional regions (optional)
    ],
    "query": str              # Description of what to look for in the selected regions
}

- Parameters:
    - region: List of dictionaries, each containing:
        - index: Integer, specifying which image to zoom in
        - bbox_2d: List of 4 integers [x1, y1, x2, y2] defining the region
    - query: String describing the search target

- Constraints:
    - At least one region must be specified
    - All coordinates must be within image boundaries
    - x1 < x2 and y1 < y2 must be satisfied

# Example:
<tool> {"region": [{"index": 0, "bbox_2d": [100, 200, 300, 400]}], "query": "Look for the red button"} </tool>
"""

SIMPLE_SYS_PROMPT = "You are a helpful assistant."

IMAGE_INDEX_PROMPT="""The index of the provided image is {index}.
"""


IMAGE_INDEX_PROMPT_V1="""The index of the provided image is {current_image_idx} (width: {width}, height: {height}).
"""

IMAGE_INDEX_PROMPT_V2="""The index of the zoom-in image is {current_image_idx} (width: {width}, height: {height}).
"""

# Answer the question using appropriate tools:
IMAGE_QUESTION_PROMPT="""This is an image indexed by 0. 
The image size: width {width}, height {height}.
"""

# multi-image may have different image size per image
MULTI_IMAGE_QUESTION_PROMPT="""These are {n_frames} images with indexed from 0 to {n_frames_1}. 
"""

VIDEO_QUESTION_PROMPT="""These are {n_frames} images with indexed from 0 to {n_frames_1}. 
All images have size: width {width}, height {height}.
"""


USER_PROMPT="""{image_instruction}
# Question: {question}

# If you need to zoom in for more details or examine specific regions, make tool call following the format:
<think> Your reasoning about where to look and why </think>
<tool> {{"region": [{{"index": int, "bbox_2d": [x1, y1, x2, y2]}}, ...], "query": str}} </tool>

# If you have enough information to answer the original question:
<think> Your reasoning here. </think>
<answer> Your final answer here. </answer>

- Note that x1, y1, x2, y2 are the coordinates of the bounding box in the specfied image by the index.
- You must strictly follow the above output format.
- In `<answer>`, provide **only** the final answer in the simplest required form:
  - For multiple-choice questions: output only the letter (e.g., `A`, `B`, `C`).
  - For yes/no questions: output only `Yes` or `No`.
  - For numerical answers: output only the number (e.g., `42`, `3.14`).
  - Do not include explanations, units, punctuation, or extra words.
"""

RESPONSE_PROMPT="""
# If you need to zoom in for more details or examine specific regions, make tool call following the format:
<think> Your reasoning about where to look and why. </think>
<tool> {{"region": [{{"index": , "bbox_2d": [x1, y1, x2, y2]}}, ...], "query": str}} </tool>

# If you have enough information to answer the original question:
<think> Your reasoning here. </think>
<answer> Your final answer here. </answer>
"""

RESPONSE_PROMPT_FINAL="""
You have reached the maximum number of iterations. **No further tool calls are allowed**.
You **must** summarize your reasoning and provide the final answer using the format below:
<think> Your reasoning here. </think>
<answer> Your final answer here. </answer>
"""

BSZ=10 # 50 reduce it if GPU OOM
MAX_IMAGES=50       # 45
SUBIMAGE_PATTERN = r".*\#\#\#\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"
TYPE_TEMPLATE = {
    "multiple choice": '\nAnswer with the option\'s letter from the given choices directly.',
    "free-form": '',
    "regression": '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).',
    "numerical": '\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).',
}

from dataclasses import dataclass
@dataclass
class ProcessData:
    index: int
    response: str
    mm_data: Dict
    original_image_url_list: List
    region_position_list: List
    query_list: List
    img_to_query_mapping: List
    finish_reason: str
    is_finished: bool
    max_pixels: int


def process_single_response(data: ProcessData):
    """处理单个响应的函数"""
    if data.is_finished is True:
        return {
            'index': data.index,
            'response': data.response,
            'finish_reason': data.finish_reason,
            'is_finished': data.is_finished,
        }
    try:
        # 解析和绘图
        parsed_output = parse_output(data.response)
        images = data.mm_data['image']
        region_position_list = data.region_position_list
        original_image_url_list = data.original_image_url_list
        zoomin_image_list, zoomin_region_position, zoomin_image_url_list, zoomin_query, valid_flag = zoomin(
            images,
            region_position_list,
            original_image_url_list,
            parsed_output,
            max_pixels=data.max_pixels
        )

        if not valid_flag:
            zoomin_feedback = zoomin_query
            return {
                'index': data.index,
                'response': data.response,
                'zoomin_valid_flag': False,
                'zoomin_feedback': zoomin_feedback,
                'finish_reason': data.finish_reason,
                'is_finished': data.is_finished,
            }
        else:
            return {
                'index': data.index,
                'response': data.response,
                'zoomin_valid_flag': True,
                'zoomin_image_list': zoomin_image_list,
                'zoomin_region_position': zoomin_region_position,
                'zoomin_image_url_list': zoomin_image_url_list, 
                'zoomin_query': zoomin_query,
                'finish_reason': data.finish_reason,
                'is_finished': data.is_finished
            }

    except Exception as e:
        print(f"Error processing response {data.index}: {str(e)}")
        traceback.print_exc()
        return None


def extract_question(text):
    # 定义正则表达式模式来匹配问题部分
    pattern = r'<image>\n(.*?) Please provide the bounding box coordinate of the region that can help you answer the question better.'
    # 执行匹配
    match = re.search(pattern, text)

    # 检查是否匹配成功，并提取问题部分
    if match:
        question = match.group(1)
        # print("匹配成功，问题部分:", question)
        return question
    else:
        return text


def save_samples_info(samples_info, save_dir):

    def get_unique_dir(base_path, prefix='generation'):
        """generate unique directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 0
        while True:
            dir_name = f"{prefix}_{timestamp}" if counter == 0 else f"{prefix}_{timestamp}_{counter}"
            full_path = os.path.join(base_path, dir_name)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    all_sample_dir = []
    for idx, sample in enumerate(samples_info):
        # 创建样本目录
        sample_dir = get_unique_dir(save_dir, f'sample_{idx}')
        os.makedirs(sample_dir, exist_ok=True)
        all_sample_dir.append(sample_dir)

        # 提取所有assistant回复
        all_responses = []
        for turn in sample['message']:
            if turn['role'] == 'assistant':
                content = turn['content']
                text = ' '.join([c.get('text', '') for c in content if c.get('type') == 'text']) if isinstance(content, list) else content
                all_responses.append(text)
        
        num_image_inputs = sample.get('num_image_inputs', 0)
        
        # 保存文本数据
        text_data = {
            'index': sample['index'],
            'message': sample['message'],
            'all_responses': all_responses,
            'final_response': all_responses[-1] if all_responses else "",
            'finish_reason': sample.get('finish_reason', 'unknown'),
            'num_image_inputs': num_image_inputs,
            'num_cropped_images': len(sample.get('multi_modal_data', {}).get('image', [])) - num_image_inputs,
        }
        
        with open(os.path.join(sample_dir, 'text_data.json'), 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
        
        # 仅保存裁剪图片（索引 >= num_image_inputs）
        images = sample.get('multi_modal_data', {}).get('image', [])
        if len(images) > num_image_inputs:
            cropped_dir = os.path.join(sample_dir, 'cropped_images')
            os.makedirs(cropped_dir, exist_ok=True)
            
            for img_idx in range(num_image_inputs, len(images)):
                if isinstance(images[img_idx], Image.Image):
                    crop_idx = img_idx - num_image_inputs
                    images[img_idx].save(os.path.join(cropped_dir, f'crop_{crop_idx}.png'))
    
    return all_sample_dir

def multi_turn_generate(inference_engine, processor, tokenizer, embed_model_tokenizer, vllm_inputs=None, sampling_params=None,  use_tqdm=False, save_dir=None, max_num_steps=10, max_pixels=None):
    
    def _get_prompts_and_indices(samples_info):
        messages, prompts, multi_modal_data, indices=[], [], [], []
        for index, info in enumerate(samples_info):
            if not info['stop'] and len(info['multi_modal_data']['image']) <= MAX_IMAGES:
                messages.append(info['message'])
                # prompts.append(info['sequence'])
                multi_modal_data.append(info['multi_modal_data'])
                indices.append(info['index'])
        return messages, prompts, multi_modal_data, indices
    
    def _get_indices(samples_info):
        indices=[]
        for index, info in enumerate(samples_info):
            if not info['stop'] and len(info['multi_modal_data']['image']) <= MAX_IMAGES:
                indices.append(info['index'])
        return  indices

    def _is_finished(finish_reason, stop_reason, response):
        if finish_reason in ['length', 'rule']:
            return True
        if finish_reason == 'stop':
            return stop_reason is None and "<answer>" in response and "</answer>" in response
        return False


    sampling_params=copy.deepcopy(sampling_params)
    new_vllm_inputs = []
    for single_vllm_input in vllm_inputs:
        # prompt = tokenizer.decode(single_vllm_input['prompt_token_ids'], skip_special_tokens=False)
        new_vllm_inputs.extend([{
            "message": copy.deepcopy(single_vllm_input['message']),
            # "prompt": prompt,
            "multi_modal_data": single_vllm_input['multi_modal_data'],
            "original_image_url_list": single_vllm_input['original_image_url_list'],
            "region_position_list": copy.deepcopy(single_vllm_input['region_position_list']),
            "query_list": copy.deepcopy(single_vllm_input['query_list']),
            "img_to_query_mapping": copy.deepcopy(single_vllm_input['img_to_query_mapping']),
        }   for _ in range(sampling_params.n)])
        
    sampling_params.n=1
    sampling_params.detokenize=True             # True convert ids to text

    samples_info = []
    for index, item in enumerate(new_vllm_inputs):
        processed_image = [fetch_image({'image': origin_image}) for origin_image in item['multi_modal_data']['image']]
        sample_info = {
            # identify
            "index": index,                 
            # content
            "message": item["message"],
            "multi_modal_data": {"image": processed_image, },
            "original_image_url_list": item['original_image_url_list'],
            "region_position_list": item['region_position_list'], 
            "query_list": item['query_list'],
            "img_to_query_mapping": item['img_to_query_mapping'],
            # state
            "stop": False,
            "finish_reason": None,
            "num_image_inputs": len(processed_image),
            # "prompt": item["prompt"],
            # "sequence": item["prompt"],
            # "response": "",
            # "processed_image_idx": [],
            # "mask_info": [],
            # "execution_pass": 0,
        }
        samples_info.append(sample_info)


    num_llm_calls_available = max_num_steps
    while num_llm_calls_available > 0:
        num_llm_calls_available = num_llm_calls_available - 1
        # messages, input_prompts, multi_modal_data, indices=_get_prompts_and_indices(samples_info)  
        indices=_get_indices(samples_info)  
        
        inputs = []
        for index in indices:
            msg = samples_info[index]['message']
            mm_data = samples_info[index]['multi_modal_data']
            prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

            encoded_inputs = embed_model_tokenizer(
                samples_info[index]['query_list'],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=50
            )
            query_ids, query_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            is_all_pad = (query_ids == embed_model_tokenizer.pad_token_id).all(dim=1)  # [B]
            query_mask[is_all_pad] = 0  # 强制全 0
            mm_data['query_ids'] = query_ids
            mm_data['query_attention_mask'] = encoded_inputs.attention_mask
            mm_data['image_to_query_mapping'] = torch.tensor(samples_info[index]['img_to_query_mapping'], dtype=torch.long)
            
            # print(f"num_llm_calls_available: {num_llm_calls_available}\nmm_data:{mm_data}")
            inputs.append({
                'prompt': prompt,
                'multi_modal_data': mm_data,
            })

        print(f"num_llm_calls_available: {num_llm_calls_available}\n")
        outputs = inference_engine.generate(prompts=inputs, sampling_params=sampling_params, use_tqdm=use_tqdm)
        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        # print("Sorted Outputs: ", sorted_outputs[0].outputs[0].text)
        responses=[x.outputs[0].text for x in sorted_outputs]
        finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]  # "stop", "length"
        stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]      # None: have EOS
        if num_llm_calls_available == 0:
            for i ,index in enumerate(indices):
                samples_info[index]['message'].append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": responses[i]}
                    ]})
                # samples_info[index]['response']+=responses[i]
                # samples_info[index]['sequence']+=responses[i]
                samples_info[index]['stop']=True
                samples_info[index]['finish_reason']=finish_reason[i]
            break

        is_finished=[_is_finished(finish_reason[i], stop_reason[i], responses[i]) for i in range(len(finish_reason))]
        if all([x for x in is_finished]): 
            for i ,index in enumerate(indices):
                samples_info[index]['message'].append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": responses[i]}
                ]})
                # samples_info[index]['response']+=responses[i]
                # samples_info[index]['sequence']+=responses[i]
                samples_info[index]['stop']=True
                samples_info[index]['finish_reason']=finish_reason[i]
            break
        
        # ----------- Parallel Process -----------
        # Prepare Data
        process_data_list = [
            ProcessData(
                index=index,
                response=responses[i],
                mm_data=samples_info[index]['multi_modal_data'],
                original_image_url_list=samples_info[index]['original_image_url_list'],
                region_position_list= samples_info[index]['region_position_list'], 
                query_list = samples_info[index]['query_list'],
                img_to_query_mapping=samples_info[index]['img_to_query_mapping'],
                finish_reason=finish_reason[i],
                is_finished=is_finished[i],                      # if is_finished == True, stop reasoning
                max_pixels=max_pixels
            )  for i, index in enumerate(indices)] 

        # 使用线程池并行处理
        # with ThreadPoolExecutor(max_workers=max(min(len(indices), os.cpu_count()//2, 64), 1) ) as executor:
        #     results = list(executor.map(process_single_response, process_data_list))
        num_workers = min(len(indices), multiprocessing.cpu_count()//4)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_response, process_data_list))

        # 更新samples_info
        for result in results:
            if result is not None:
                index = result['index']
                samples_info[index]['message'].append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": result['response']}
                ]})
                # samples_info[index]['response'] += result['response']
                samples_info[index]['stop'] = result['is_finished']
                samples_info[index]['finish_reason'] = result['finish_reason']

                if result['is_finished'] is False:
                    content_feedback = []
                    zoomin_valid_flag = result['zoomin_valid_flag']
                    if not zoomin_valid_flag:
                        zoomin_feedback = result['zoomin_feedback']
                        content_feedback.append({
                            "type": "text", 
                            "text": "<tool_response>" + f"[Error] Zoom-in failed: {zoomin_feedback}\n"
                        })
                        content_feedback.append({"type": "text", "text": RESPONSE_PROMPT_FINAL + "</tool_response>"})

                    else:
                        current_img_idx  = len(samples_info[index]['multi_modal_data']['image'])
                        rest_image_count = max(MAX_IMAGES -  current_img_idx, 0)
                        if rest_image_count == 0:
                                print(f"⚠️ Sample {index}: Reached MAX_IMAGES={MAX_IMAGES}, stopping zoomin")
                                content_feedback.append({"type": "text", "text": "<tool_response>" + RESPONSE_PROMPT_FINAL + "</tool_response>"})
                        else:
                        
                            zoomin_image_list = result['zoomin_image_list'][:rest_image_count]
                            zoomin_region_position = result['zoomin_region_position'][:rest_image_count]
                            zoomin_image_url_list = result['zoomin_image_url_list'][:rest_image_count]
                            zoomin_query = result['zoomin_query']
                        
                            samples_info[index]['multi_modal_data']['image'].extend(zoomin_image_list)
                            samples_info[index]['region_position_list'].extend(zoomin_region_position)
                            samples_info[index]['original_image_url_list'].extend(zoomin_image_url_list)
                            samples_info[index]['query_list'].append(zoomin_query)
                            query_idx = len(samples_info[index]['query_list']) - 1
                            samples_info[index]['img_to_query_mapping'].extend([query_idx for _ in range(len(zoomin_image_list))])        # new

                            content_feedback.append({"type": "text", "text": "<tool_response>"})
                            for zoomin_image, region_position in zip(zoomin_image_list, zoomin_region_position):
                                zoomin_width, zoomin_height = zoomin_image.size
                                content_feedback.append({"type": "image"})
                                content_feedback.append({"type": "text", "text": IMAGE_INDEX_PROMPT_V2.format(current_image_idx=current_img_idx, width=zoomin_width, height=zoomin_height)})
                                current_img_idx += 1

                            if num_llm_calls_available <= 1 or current_img_idx >= MAX_IMAGES:
                                # tool call ends or can't accept new images
                                content_feedback.append({"type": "text", "text": RESPONSE_PROMPT_FINAL + "</tool_response>"})
                            else:
                                content_feedback.append({"type": "text", "text": RESPONSE_PROMPT + "</tool_response>"})

                    samples_info[index]['message'].append(
                        {"role": "user", "content": content_feedback}
                    )

    batch_messages = [sample['message'] for sample in samples_info]
    if save_dir:
        all_sample_dir = save_samples_info(samples_info, save_dir)
        return batch_messages, all_sample_dir
    return batch_messages


def parse_dialog(serialized_content):
    # 分割对话内容
    segments = re.split(r'<\|im_start\|>|<\|im_end\|>', serialized_content)
    segments = [s for s in segments if s]  # 移除空字符串，但不strip
    
    conversations = []
    current_role = None
    current_content = []
    
    # 系统提示的处理
    system_content = None
    if segments[0].startswith('system'):
        system_content = segments[0].replace('system\n\n', '', 1)  # 只替换第一次出现
        segments = segments[1:]
    
    # 初始化对话列表
    if system_content:
        conversations.append({
            "role": "system",
            "content": system_content
        })

    # 处理用户和助手的对话
    for segment in segments:
        if segment.startswith('user'):
            # 提取图像标记和文本
            has_vision = '<|vision_start|><|image_pad|><|vision_end|>' in segment
            text = segment.replace('user\n', '', 1)  # 只替换第一次出现
            # text = text.replace('<|vision_start|><|image_pad|><|vision_end|>\n', '', 1)               # keep <|vision_start|><|image_pad|><|vision_end|>
            
            content = []
            if has_vision:
                content.append({
                    "type": "image",
                    "image": "image_path",
                    "nframes": "args.max_frames",
                    "max_pixels": args.max_pixels
                })
            content.append({
                "type": "text",
                "text": text
            })
            
            conversations.append({
                "role": "user",
                "content": content
            })
        elif segment.startswith('assistant'):
            text = segment.replace('assistant\n', '', 1)  # 只替换第一次出现
            conversations.append({
                "role": "assistant",
                "content": text
            })
    
    return conversations


def ensure_image_url(image: str) -> str:
    if os.path.exists(image):
        return image
    raise ValueError(f"Invalid image: {image}")


def extract_image_path(contents: list[dict[str, str]]):
    user_image_path_list = []
    content_history = copy.deepcopy(contents)
    for rou in content_history:
        if rou["type"] != "image":
            continue
        if "value" in rou:
            user_image_path_list.append(rou["value"])
        elif "image" in rou:
            user_image_path_list.append(rou["image"])
    return user_image_path_list


def generate_prompt_qa(user_question, user_image_path_list,  max_pixels, min_pixels):
    # Construct the prompt based on the given requirements
    try:
        fetch_image_dict = {"image": user_image_path_list[-1]}
        if max_pixels is not None:
            fetch_image_dict['max_pixels'] = max_pixels
        if min_pixels is not None:
            fetch_image_dict['min_pixels'] = min_pixels
        img = fetch_image(fetch_image_dict)
        width, height = img.size
    except Exception:
        width, height = None, None
        
    if len(user_image_path_list) > 1:
        # each video frame has the same size
        image_instruction =  VIDEO_QUESTION_PROMPT.format(n_frames=len(user_image_path_list), 
        width=width, height=height, n_frames_1=len(user_image_path_list)-1) 
    else:
        image_instruction =  IMAGE_QUESTION_PROMPT.format(width=width, height=height) 

    prompt =  USER_PROMPT.format(image_instruction=image_instruction, question=user_question)
    return prompt


def prepare_content(
    inputs: list[dict[str, str]], min_pixels=None, max_pixels=None,
) -> list[dict[str, str]]:
    """
    inputs list[dict[str, str]], each dict has keys: ['type', 'value']
    """
    user_image_path_list = extract_image_path(inputs)

    num_images=0
    single_image_max_pixels = None
    for s in inputs:
        if s["type"] == "image":
            num_images = num_images + 1
    if max_pixels is not None:
        single_image_max_pixels = max_pixels / num_images
    content = []
    image_idx = 0
    for s in inputs:
        if s["type"] == "image":
            item = {"type": "image", "image": ensure_image_url(s["value"])}
            if min_pixels is not None:
                item["min_pixels"] = min_pixels
            if single_image_max_pixels is not None:
                item["max_pixels"] = single_image_max_pixels
        elif s["type"] == "text":
            item = {"type": "text",
            "text": generate_prompt_qa(
                        s['value'], 
                        user_image_path_list,
                        max_pixels=single_image_max_pixels,
                        min_pixels=min_pixels)}
        else:
            raise ValueError(f"Invalid message type: {s['type']}, {s}")
        content.append(item)

        if s["type"] == 'image':
            index_prompt = IMAGE_INDEX_PROMPT.format(index=image_idx)
            item = {"type": "text", "text": index_prompt}
            content.append(item)
            image_idx = image_idx + 1

    return content


def eval_model(args):
    # Model
    model_path = args.model_path
    model_name = args.model_name

    print(f"Loading from {model_path}")
    # llm = LLM(
    #         model=model_path, 
    #         dtype="bfloat16", 
    #         tensor_parallel_size=torch.cuda.device_count(),
    #         limit_mm_per_prompt={"image": 62, "video": 0},
    #         gpu_memory_utilization=0.85,                   # default 0.9
    #         enable_prefix_caching=True,                     # cache,
    #         enforce_eager=True
    #       )
    llm = LLM(
            model=model_path, 
            dtype="bfloat16", 
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": 52, "video": 0},      # video: 0. unsupported for video
            gpu_memory_utilization=0.60,                   # default 0.9
            enable_prefix_caching=True,                    # cache
            enforce_eager=True,       # 禁用 cudagraph
            trust_remote_code=True,
            disable_mm_preprocessor_cache=True,             # 禁用mm cache
        )
    print("Load vLLM Done!")
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=16384,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    embedding_model_path = "Qwen/Qwen3-Embedding-0.6B"
    embed_model_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)

    file_path = args.input_file
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    st, ed = (len(data)*args.split)//args.all, (len(data)*(args.split+1))//args.all
    # st, ed = 10, 50
    print(f"{len(data)} lines found, generating from {st} to {ed}")
    print("Data: ", len(data))
    data = data[st:ed]
    messages = []
    for xidx, x in enumerate(data[:]):
        # 1. prompt build
        prompt = x['question']
        ptype = x['problem_type'].lower()
        if ptype == 'multiple choice' and x.get('options', None) is not None:
            prompt += '\n' + '\n'.join(x['options'])
        prompt += TYPE_TEMPLATE.get(ptype, "")

        # 2. simple message build
        simple_message = []
        if  isinstance(x['images'], str):
            x['image_path'] = [x['image_path']]
        for img_path in x['images']:
            full_path = os.path.join(args.image_folder, img_path) if args.image_folder else img_path
            simple_message .append({'type': 'image', 'value': full_path})
        simple_message.append({'type': 'text', 'value': prompt})

        # 3. reasoning message build
        message = []
        message.append({"role": "system", "content": REASONING_SYSTEM_PROMPT})
        message.append(
            {"role": "user", "content": prepare_content(simple_message, min_pixels=args.min_pixels, max_pixels=args.max_pixels)}
        )
        messages.append(message)

    # if args.all > 1:
        # 分split
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset}_{args.split}_{args.all}") 
    # else:
    #     output_dir = args.output_dir 
    save_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process data in batches
    start_idx = 0
    output_file_path = f"{output_dir}/{args.model_name}_{args.dataset}_results.jsonl"
    # 如果需要覆盖，只删除目标文件（而非整个目录！）
    if args.over_write and os.path.exists(output_file_path):
        os.remove(output_file_path)
        print(f"Removed existing file: {output_file_path}")
    if os.path.exists(output_file_path):
        mode = "a"
        with open(output_file_path) as fin:
            for line in fin:
               start_idx += 1
    else:
        mode = "w"
    print("Output Dir: ", output_dir)

    with open(output_file_path, mode, encoding="utf-8") as fout:
        print("Message Example:", messages[0])
        print(f"Start from the {start_idx} example")
        for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
            # if i < 780:
            #     continue
            batch_messages = messages[i:i + BSZ]
            prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_num = []
            for msg in batch_messages:
                current_image_num = 0
                for turn in msg:
                    if isinstance(turn["content"], list):
                        for turn_content in turn["content"]:
                            if turn_content["type"] == "image":
                                current_image_num += 1
                image_num.append(current_image_num)
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            image_idx = 0
            video_idx = 0
            llm_inputs = []
            # breakpoint()
            for idx, (prompt, msg) in enumerate(zip(prompts, batch_messages)):
                mm_type = batch_messages[idx][1]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = []
                    for current_idx in range(image_num[idx]):
                        width, height = image_inputs[image_idx].size
                        sample_mm_data["image"].append(image_inputs[image_idx])             # resize(, Image.Resampling.LANCZOS)
                        image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = [video_inputs[video_idx]]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                # print(sample_mm_data)
                original_image_url_list = extract_image_path(msg[1]['content'])
                region_position_list = [[0., 0., 1., 1.] for _ in range(len(original_image_url_list))]
                query_list = ["" for _ in range(len(original_image_url_list))]
                img_to_query_mapping = list(range(len(original_image_url_list))) 

                llm_inputs.append({
                    "message": msg,
                    # "prompt": prompt,
                    # "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                    "original_image_url_list": original_image_url_list,
                    "region_position_list": region_position_list,
                    "query_list": query_list,
                    "img_to_query_mapping": img_to_query_mapping
                })
                if "image" not in sample_mm_data:
                    print(f"Id-{idx} not have image!")
                    breakpoint()
                    continue
            if image_inputs is not None:
                assert image_idx == len(image_inputs), f"Image index mismatch: {image_idx} != {len(image_inputs)}"
            if video_inputs is not None:
                assert video_idx == len(video_inputs), f"Video index mismatch: {video_idx} != {len(video_inputs)}"


            # print(f"message: {msg}")

            # print(f"prompts[0]: {prompts[0]}")
            if i < 0:
                batch_conversations = multi_turn_generate(llm, processor, tokenizer, embed_model_tokenizer, vllm_inputs=llm_inputs, sampling_params=sampling_params, save_dir=save_dir, max_num_steps=args.round, max_pixels=args.max_pixels)
                batch_conversations, all_sample_dir = batch_conversations
            else:
                batch_conversations = multi_turn_generate(llm, processor, tokenizer, embed_model_tokenizer,vllm_inputs=llm_inputs, sampling_params=sampling_params, save_dir=None, max_num_steps=args.round, max_pixels=args.max_pixels)
                all_sample_dir = [None] * len(batch_conversations)

            # batch_conversations = [parse_dialog(sequence) for sequence in batch_sequences]
            print(f"Processed batch {(i)//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}. ")
            for input_example, model_output, sample_dir in zip(data[i:i + BSZ], batch_conversations, all_sample_dir):
                result = input_example.copy()
                result['conversations'] = model_output
                result['response'] = extract_answer(model_output[-1]['content'])
                result['model_id'] = model_name
                result['sample_dir'] = sample_dir
                # breakpoint()

                fout.write(
                    json.dumps(result)
                    + "\n"
                )
                fout.flush()


if __name__ == "__main__":

    my_seed = 1234         
    set_seed(my_seed)


    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-type", type=str, required=True, help="")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the question file")
    parser.add_argument("--output-dir", type=str, default="./result")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=256*28*28)
    parser.add_argument("--over_write", type=int, default=0, help="Whether to overwrite the output directory")
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--all", type=int, default=1)
    parser.add_argument("--round", type=int, default=10)
    args = parser.parse_args()
    if args.image_folder == "None":
        args.image_folder = ""
    eval_model(args)
    