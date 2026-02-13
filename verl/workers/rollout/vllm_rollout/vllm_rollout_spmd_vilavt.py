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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import os
os.environ['VLLM_ALLREDUCE_USE_SYMM_MEM'] = '0'
from contextlib import contextmanager
from typing import Any, List, Union, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, AutoProcessor, AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig

import copy
import pdb
from PIL import Image
from qwen_vl_utils import fetch_image
from ....models.transformers.qwen2_vl import get_rope_index
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
# from ....models.transformers.qwen2_5_vl import get_rope_index

from  ....utils.vilavt_utils import (zoomin, 
parse_output, 
parse_dialogue,
REASONING_SYSTEM_PROMPT,
SIMPLE_SYS_PROMPT,
IMAGE_INDEX_PROMPT,
IMAGE_INDEX_PROMPT_V2,
IMAGE_QUESTION_PROMPT,
MULTI_IMAGE_QUESTION_PROMPT,
VIDEO_QUESTION_PROMPT,
USER_PROMPT,
RESPONSE_PROMPT,
RESPONSE_PROMPT_FINAL,
)
import json
MAX_IMAGES=52

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

# 安全转换函数
def to_list_safe(data):
    """安全地将数据转换为 list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return data
    else:
        # 如果是单个值，包装成列表
        return [data]

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
                # 'finish_reason': data.finish_reason,
                # 'is_finished': data.is_finished,
                'finish_reason': "Zoomin failed",
                'is_finished': True,

            }
        elif len(data.mm_data['image']) + len(zoomin_image_list) > MAX_IMAGES:
            return {
                'index': data.index,
                'response': data.response,
                'zoomin_valid_flag': False,
                'zoomin_feedback': "TooManyImages",
                # 'finish_reason': data.finish_reason,
                # 'is_finished': data.is_finished,
                'finish_reason': "TooManyImages",
                'is_finished': True,

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


def check_repetition(allindex, bbox_list_origin, movement_list_origin):
    # print(allindex)
    # print(bbox_list_origin)
    # print(movement_list_origin)
    for cnt, tmp_index in enumerate(allindex):
        for bbox_list in list(bbox_list_origin.values()):
            for bbox in bbox_list:
                if bbox in allindex[tmp_index]["bbox_list"]:
                    return True
        for movement_list in list(movement_list_origin.values()):
            for movement in movement_list:
                if movement in allindex[tmp_index]["movement_list"]:
                    return True
    return False


def save_samples_info(samples_info, save_dir):

    def get_unique_dir(base_path, prefix='generation'):
        """generate unique dirctory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 0
        while True:
            if counter == 0:
                dir_name = f"{prefix}_{timestamp}"
            else:
                dir_name = f"{prefix}_{timestamp}_{counter}"

            full_path = os.path.join(base_path, dir_name)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    all_sample_dir = []
    for idx, sample in enumerate(samples_info):
        # 为每个样本创建子目录
        sample_dir = get_unique_dir(save_dir, f'sample')
        os.makedirs(sample_dir, exist_ok=True)
        all_sample_dir.append(sample_dir)

        # 保存文本信息
        text_data = {
            'prompt': sample['prompt'],
            'sequence': sample['sequence'],
            'response': sample['response'],
            'finish_reason': sample['finish_reason'],
            'execution_pass': sample['execution_pass']
        }
        
        with open(os.path.join(sample_dir, 'text_data.json'), 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
        
        # 保存图片
        if 'multi_modal_data' in sample and 'image' in sample['multi_modal_data']:
            images_dir = os.path.join(sample_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            for img_idx, img in enumerate(sample['multi_modal_data']['image']):
                if isinstance(img, Image.Image):
                    img_path = os.path.join(images_dir, f'image_{img_idx}.png')
                    img.save(img_path)
    return all_sample_dir


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


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer, processor: AutoProcessor):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor                  # add processor
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if processor is not None:  # only VLMs have processor
            vllm_init_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images > 0:
                vllm_init_kwargs["limit_mm_per_prompt"] =  {"image": config.limit_images, "video": 0}  # disable video

        print(f"[vLLMRollout Init] VLLM_ALLREDUCE_USE_SYMM_MEM={os.getenv('VLLM_ALLREDUCE_USE_SYMM_MEM')}")
        print(f"[vLLMRollout Init] Tensor Parallel Size={config.tensor_parallel_size}")
        print("=========== Start loading inference engine!  ===========")
        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            # trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            # seed=config.seed,
            max_model_len=int((config.prompt_length + config.response_length)*1.5),
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **vllm_init_kwargs,
        )
        print(" =========== Load inference engine Done! ===========")
        if "baseline" in model_path:
            self.use_baseline = True
            print(f"use_baseline in model_path: {model_path}")
        else:
            self.use_baseline = False

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)
        # sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        sampling_kwargs = {"max_tokens": config.single_turn_response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        embedding_model_path = "Qwen/Qwen3-Embedding-0.6B"
        self.embed_model_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _get_multi_turn_mask(self, response_tokens):
        """
        生成多轮对话的attention mask，mask掉所有特殊标记和提示部分
        
        Args:
            response_tokens: 包含多轮对话的token序列
            
        Returns:
            attention_mask: 与response_tokens同样大小的mask，只保留助手的响应内容
        """
        
        # 获取特殊token的id
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = self.tokenizer.convert_tokens_to_ids("user")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        pad_id = self.tokenizer.pad_token_id
        newline_id = 198

        attention_mask = torch.zeros_like(response_tokens)  # 初始化全为0
        current_pos = 0
        in_assistant_response = True  # 初始状态为True，因为从assistant响应开始
        while current_pos < len(response_tokens):
            if response_tokens[current_pos] == im_end_id:
                # 遇到im_end_id，切换状态
                in_assistant_response = False
                current_pos += 1
                continue
                
            if (current_pos + 2 < len(response_tokens) and 
                response_tokens[current_pos] == im_start_id and 
                response_tokens[current_pos + 1] == assistant_id and
                response_tokens[current_pos + 2] == newline_id):
                # 找到新的assistant响应开始（包括换行符）
                in_assistant_response = True
                current_pos += 3  # 跳过im_start, assistant和换行符
                continue
                
            if in_assistant_response and response_tokens[current_pos] != pad_id:
                # 在assistant响应内容中，且不是padding
                attention_mask[current_pos] = 1
            current_pos += 1

        return attention_mask

    def decode_masked_tokens(self, input_ids, mask, prompt_len):
        """
        对mask非0部分进行detokenize
        
        Args:
            input_ids: 输入token序列
            mask: attention mask
            prompt_len: prompt长度
        """
        # 获取response部分的tokens和mask
        response_tokens = input_ids[prompt_len:self.config.response_length]
        response_mask = mask.bool()  # 转换为布尔掩码
        
        # 收集所有需要decode的token段 
        valid_segments = []
        current_segment = []
        
        for token, is_valid in zip(response_tokens, response_mask):
            if is_valid:
                current_segment.append(token.item())
            elif current_segment:  # 当前段结束
                valid_segments.append(current_segment)
                current_segment = []
        
        if current_segment:  # 处理最后一个段
            valid_segments.append(current_segment)
        
        # 对每个有效段进行decode
        decoded_segments = []
        for segment in valid_segments:
            decoded_text = self.tokenizer.decode(segment, skip_special_tokens=True)
            decoded_segments.append(decoded_text)
        
        return decoded_segments
    
    def decode_masked_tokens_2(self, input_ids, mask, prompt_len):
        """
        将mask==1部分替换为pad_token，其余部分保留
        
        Args:
            input_ids: 输入token序列
            mask: attention mask
            prompt_len: prompt长度
        """
        # 获取response部分的tokens和mask
        response_tokens = input_ids[prompt_len:self.config.response_length].clone()  # 创建副本
        response_mask = mask.bool()
        
        # 获取pad token id
        pad_token_id = self.tokenizer.pad_token_id
        
        # 将mask==1的部分替换为pad_token
        response_tokens[response_mask] = pad_token_id
        
        # decode整个序列
        decoded_text = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
        
        # 同时decode原始序列用于对比
        original_text = self.tokenizer.decode(
            input_ids[prompt_len:self.config.response_length], 
            skip_special_tokens=False
        )
        
        return {
            'masked_text': decoded_text,
            'original_text': original_text
        }


    def check_token_length(self, sequence, images):
        """检查当前序列的总token长度"""
        # 计算文本token
        text_tokens = len(self.tokenizer.encode(sequence, add_special_tokens=False))
        # 估算图片token (根据实际模型调整)
        image_tokens = len(images) * 576  # 假设每张图片大约占用576个token
        return text_tokens + image_tokens

    def _get_indices(self, samples_info):
        indices=[]
        for index, info in enumerate(samples_info):
            if not info['stop'] and len(info['multi_modal_data']['image']) <= MAX_IMAGES:
                indices.append(info['index'])
        return  indices

    def _multi_turn_generate(self, vllm_inputs=None,   sampling_params=None,  use_tqdm=False, save_dir=None, max_num_steps=10):
    
        def _is_finished(finish_reason, stop_reason, response):
            if finish_reason in ['length', 'rule']:
                return True
            if finish_reason == 'stop':
                return stop_reason is None and "<answer>" in response and "</answer>" in response
            return False

        sampling_params=copy.deepcopy(sampling_params)
        new_vllm_inputs = []
        for single_vllm_input in vllm_inputs:
            new_vllm_inputs.extend([{
                "message": copy.deepcopy(single_vllm_input['message']),
                # "prompt": prompt,
                "multi_modal_data": single_vllm_input['multi_modal_data'],
                "original_image_url_list": copy.deepcopy(single_vllm_input['original_image_url_list']),
                "region_position_list": copy.deepcopy(single_vllm_input['region_position_list']),
                "query_list": copy.deepcopy(single_vllm_input['query_list']),
                "img_to_query_mapping": copy.deepcopy(single_vllm_input['img_to_query_mapping']),
                "max_pixels": single_vllm_input['max_pixels'],
                "max_num_steps": single_vllm_input["max_num_steps"]
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
                "multi_modal_data": {"image": processed_image},
                "original_image_url_list": item['original_image_url_list'],
                "region_position_list": item['region_position_list'], 
                "query_list": item['query_list'],
                "img_to_query_mapping": item['img_to_query_mapping'],
                # state
                "stop": False,
                "finish_reason": None,
                "max_pixels": item["max_pixels"],
                "max_num_steps": item["max_num_steps"]
            }
            samples_info.append(sample_info)

        # num_llm_calls_available=copy.deepcopy(self.config.num_llm_calls_available) - 1 
        # print(f"num_llm_calls_available: {num_llm_calls_available}\n")
        num_llm_calls_available_list = [
          sample.get('max_num_steps', 5) or 5  # 如果 round 为 None 或 0，使用 5
          for sample in samples_info]

        while any(num_calls > 0 for num_calls in num_llm_calls_available_list):
            print(f"num_llm_calls_available_list: {num_llm_calls_available_list}")
            indices=self._get_indices(samples_info)  
            for i in range(len(num_llm_calls_available_list)):      # 更新每个样本的调用次数
                if num_llm_calls_available_list[i] > 0:
                    num_llm_calls_available_list[i] -= 1
            
            batch_inputs = []
            for index in indices:
                msg = samples_info[index]['message']
                mm_data = samples_info[index]['multi_modal_data']
                prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

                encoded_inputs = self.embed_model_tokenizer(
                    samples_info[index]['query_list'],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=50
                )
                query_ids, query_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
                is_all_pad = (query_ids == self.embed_model_tokenizer.pad_token_id).all(dim=1)  # [B]
                query_mask[is_all_pad] = 0  # 强制全 0
                if self.use_baseline is False:
                    mm_data['query_ids'] = query_ids
                    mm_data['query_attention_mask'] = query_mask
                    mm_data['image_to_query_mapping'] = torch.tensor(samples_info[index]['img_to_query_mapping'], dtype=torch.long)
                # print(f"num_llm_calls_available: {num_llm_calls_available}\nmm_data:{mm_data}")
                batch_inputs.append({
                    'prompt': prompt,
                    'multi_modal_data': mm_data,
                })
            
            # print(f"Inputs[0]: {batch_inputs[0]}")
            # print(f"multi modal data: {[len(x['multi_modal_data']) for x in batch_inputs]}")
            outputs = self.inference_engine.generate(prompts=batch_inputs, sampling_params=sampling_params, use_tqdm=use_tqdm)
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            # print("Sorted Outputs: ", sorted_outputs[0].outputs[0].text)
            responses=[x.outputs[0].text for x in sorted_outputs]
            finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]  # "stop", "length"
            stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]      # None: have EOS
            if all(num_calls == -1 for num_calls in num_llm_calls_available_list):
                for i ,index in enumerate(indices):
                    samples_info[index]['message'].append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": responses[i]}
                        ]})
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
                    is_finished=is_finished[i], # if is_finished == True, stop reasoning
                    max_pixels=samples_info[index]['max_pixels']
                )  for i, index in enumerate(indices)] 

            # 使用线程池并行处理
            num_workers = min(len(indices), multiprocessing.cpu_count()//4)
            # with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
                        # zoomin_valid_flag = result['zoomin_valid_flag']
                        # if not zoomin_valid_flag:
                        #     zoomin_feedback = result['zoomin_feedback']
                        #     content_feedback.append({
                        #         "type": "text", 
                        #         "text": "<tool_response>" + f"[Error] Zoom-in failed: {zoomin_feedback}\n"
                        #     })
                        #     content_feedback.append({"type": "text", "text": RESPONSE_PROMPT_FINAL + "</tool_response>"})

                        # else:
                        current_img_idx  = len(samples_info[index]['multi_modal_data']['image'])
                        rest_image_count = max(MAX_IMAGES -  current_img_idx, 0)
                            # if rest_image_count == 0:
                            #     print(f"⚠️ Sample {index}: Reached MAX_IMAGES={MAX_IMAGES}, stopping zoomin")
                            #     content_feedback.append({"type": "text", "text": "<tool_response>" + RESPONSE_PROMPT_FINAL + "</tool_response>"})
                            # else:
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

                        if num_llm_calls_available_list[index] <= 1 or current_img_idx >= MAX_IMAGES:
                            # tool call ends or can't accept new images
                            content_feedback.append({"type": "text", "text": RESPONSE_PROMPT_FINAL + "</tool_response>"})
                        else:
                            content_feedback.append({"type": "text", "text": RESPONSE_PROMPT + "</tool_response>"})

                        samples_info[index]['message'].append(
                            {"role": "user", "content": content_feedback}
                        )

        batch_messages = [sample['message'] for sample in samples_info]
        image_inputs = []
        image_to_query_mapping_list = []
        query_list = []
        for idx, sample_info in enumerate(samples_info):
            num_images = len(sample_info['multi_modal_data']['image'])
            if num_images > MAX_IMAGES:
                print(f"⚠️⚠️⚠️ CRITICAL: Sample {idx} has {num_images} images, exceeding MAX_IMAGES={MAX_IMAGES}")
                sample_info['multi_modal_data']['image'] = sample_info['multi_modal_data']['image'][:MAX_IMAGES] # 截断到 MAX_IMAGES
                sample_info['img_to_query_mapping'] = sample_info['img_to_query_mapping'][:MAX_IMAGES]
                if len(sample_info['img_to_query_mapping']) > 0:
                    max_query_num = max(sample_info['img_to_query_mapping']) + 1
                    sample_info['query_list'] = sample_info['query_list'][:max_query_num]
            image_inputs.append(sample_info['multi_modal_data']['image'])
            image_to_query_mapping_list.append(sample_info['img_to_query_mapping'])
            query_list.append(sample_info['query_list'])

        if save_dir:
            all_sample_dir = save_samples_info(samples_info, save_dir)
            # return batch_messages, all_sample_dir
        return batch_messages, image_inputs, image_to_query_mapping_list, query_list


    def _mask(self, ):
        pass

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (rollout_bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for msg, multi_modal_data, max_pixels, max_num_steps, image_paths in zip(
                # non_tensor_batch.pop("raw_prompt_ids"), 
                non_tensor_batch.pop("message"), 
                non_tensor_batch.pop("multi_modal_data"), 
                non_tensor_batch.pop("max_pixels"),             # origin: pop
                non_tensor_batch.pop("max_num_steps"),
                non_tensor_batch.pop("image_path")
            ):
                
                msg = to_list_safe(msg)
                original_image_url_list = to_list_safe(image_paths)
                region_position_list = [[0., 0., 1., 1.] for _ in range(len(original_image_url_list))]
                query_list = ["" for _ in range(len(original_image_url_list))]
                img_to_query_mapping = list(range(len(original_image_url_list))) 
                
                # print(f"Message: {msg}")
                vllm_inputs.append({
                    "message": msg,
                    #   "prompt_token_ids": list(raw_prompt_ids), 
                    "multi_modal_data": multi_modal_data, 
                    "max_num_steps": max_num_steps,
                    "max_pixels": max_pixels,
                    "original_image_url_list": original_image_url_list,
                    "region_position_list": region_position_list,
                    "query_list": query_list,
                    "img_to_query_mapping": img_to_query_mapping
                })
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        with self.update_sampling_params(**prompts.meta_info):
            batch_conversations, image_inputs, img_to_query_list, all_query_list = self._multi_turn_generate(vllm_inputs=vllm_inputs, sampling_params=self.sampling_params, save_dir=None, max_num_steps=10) # save_dir="./tmp_trace/20250512"
            sequences = self.processor.apply_chat_template(batch_conversations, add_generation_prompt=False, tokenize=False)
            # print(f"sequneces: {sequences[:1]}")

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)           # repeat tensor or list at dim=0
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
        
        
        non_tensor_batch["raw_prompt_ids"] =  [self.tokenizer.encode(sequence, add_special_tokens=False)[:self.config.prompt_length+self.config.response_length] for sequence in sequences] # raw prompt+response ids
        valid_prompt_len = torch.sum(attention_mask, dim=-1)
        response_ids = []
        response_mask = []
        response_position_ids = []
        model_inputs = []
        multi_turn_mask = []
        # sequence_ids = []
        for idx, prompt_len in enumerate(valid_prompt_len):
            inputs = self.processor(text=sequences[idx], 
                                    images=image_inputs[idx], 
                                    add_special_tokens=False,              # whether add special tokens like <|im_start|><||>
                                    # padding='max_length',                   # ['longest', 'max_length', 'do_not_pad']
                                    # max_length=self.config.prompt_length + self.config.response_length,
                                    return_tensors="pt")                    # for transformers
            
            check_range = prompt_len + self.config.response_length
            num_vision_start = torch.sum(inputs['input_ids'][0][:check_range] == 151652).item()
            num_vision_end = torch.sum(inputs['input_ids'][0][:check_range] == 151653).item()
            expected_images = len(image_inputs[idx])
            need_truncation = (num_vision_start != expected_images or  num_vision_end != expected_images)       # 判断是否需要截断
            if need_truncation:
                print(f"⚠️ Batch {idx}: Truncation needed")
                print(f"   Expected images: {expected_images}")
                print(f"   Found vision_start: {num_vision_start}, vision_end: {num_vision_end}")
                print(f"   Original turns: {len(batch_conversations[idx])}")
                batch_conversations[idx] = batch_conversations[idx][:3]                 # only keep system_prompt, user prompt and first assistant prompt
                sequences[idx] = self.processor.apply_chat_template(batch_conversations[idx], add_generation_prompt=False, tokenize=False)
                num_images = sequences[idx].count("<|vision_end|>")
                image_inputs[idx] = image_inputs[idx][:num_images]
                img_to_query_list[idx] = img_to_query_list[idx][:num_images]
                unique_query_indices = sorted(set(img_to_query_list[idx]))
                all_query_list[idx] = all_query_list[idx][:max(unique_query_indices)+1] 
                inputs = self.processor(text=sequences[idx], 
                                    images=image_inputs[idx], 
                                    add_special_tokens=False,              # whether add special tokens like <|im_start|><||>
                                    # padding='max_length',                   # ['longest', 'max_length', 'do_not_pad']
                                    # max_length=self.config.prompt_length + self.config.response_length,
                                    return_tensors="pt")            


            new_position_ids = get_rope_index(
                self.processor,
                input_ids=inputs['input_ids'][0],
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=inputs['attention_mask'][0],
            )  # (3, seq_length)
            assert torch.sum(input_ids[idx][-prompt_len:].cpu() == inputs['input_ids'][0][:prompt_len].cpu()) == prompt_len, \
                f"Input IDs mismatch at batch index {idx}. Former: {torch.sum(input_ids[idx][-prompt_len:].cpu() == inputs['input_ids'][0][:prompt_len].cpu())}, Later: {prompt_len}"
            
            assert torch.sum(attention_mask[idx][-prompt_len:].cpu() == inputs['attention_mask'][0][:prompt_len].cpu()) == prompt_len, \
                f"Attention mask mismatch at batch index {idx}"
            assert torch.sum(position_ids[idx, :, -prompt_len:].cpu()== new_position_ids[: ,:prompt_len].cpu()) == prompt_len * 3, \
                f"Attention mask mismatch at batch index {idx}"

            # assert torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151652) == len(image_inputs[idx]), \
            #     f"Number of <|vision_start|> {torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151652)} != Image Inputs {len(image_inputs[idx])}"
            # assert torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151653) == len(image_inputs[idx]), \
            #     f"Number of <|vision_end|> {torch.sum(inputs['input_ids'][0][:prompt_len+self.config.response_length] == 151653)} != Image Inputs {len(image_inputs[idx])}"
            # print(f"Number of <|vision_start|> {sum([x==151652 for x in non_tensor_batch['raw_prompt_ids'][idx]])}, Number of <|vision_end|> {sum([x==151653 for x in non_tensor_batch['raw_prompt_ids'][idx]])},  Image Inputs {len(image_inputs[idx])}")
            
            response_ids.append(inputs['input_ids'][0][prompt_len: prompt_len+self.config.response_length])
            response_mask.append(inputs['attention_mask'][0][prompt_len: prompt_len+self.config.response_length])
            pad_position_ids = VF.pad_sequence_to_length(new_position_ids[:, prompt_len: prompt_len + self.config.response_length], max_seq_len=self.config.response_length, pad_token_id=0, left_pad=False).to(input_ids.device)        # (3, max_length)
            response_position_ids.append(pad_position_ids)
            tmp_multi_turn_mask = self._get_multi_turn_mask(inputs['input_ids'][0][prompt_len: prompt_len+self.config.response_length])
            # if idx %4==0:
            #     decode_segements= self.decode_masked_tokens(inputs['input_ids'][0], tmp_multi_turn_mask, prompt_len )
            #     print(f"decode_segements: {decode_segements}")
            multi_turn_mask.append(tmp_multi_turn_mask)

            inputs.pop('input_ids')
            inputs.pop('attention_mask')

            encoded_inputs = self.embed_model_tokenizer(
                all_query_list[idx],  # ✅ 使用从生成中获取的 query_list
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=50
                )
            query_ids = encoded_inputs.input_ids
            query_attention_mask = encoded_inputs.attention_mask
            is_all_pad = (query_ids == self.embed_model_tokenizer.pad_token_id).all(dim=1)  # [B]
            query_attention_mask[is_all_pad] = 0  # 强制全 0

            if self.use_baseline is False:
                inputs["image_to_query_mapping"] = torch.tensor(img_to_query_list[idx], dtype=torch.long)
                inputs["query_ids"] = query_ids
                inputs["query_attention_mask"] = query_attention_mask
            model_inputs.append(dict(inputs))               # convert transformers.feature_extraction_utils.BatchFeature to dict

        print("Response Ids: ", [len(x) for x in response_ids])
        for idx, x in enumerate(response_ids):
            if len(x) > self.config.response_length:
                print(">response_length X:", x)
                print("Sequence: ", sequences[idx])
                print("\n")
        response_ids = VF.pad_2d_list_to_length(response_ids, self.pad_token_id, max_length=self.config.response_length).to(input_ids.device)                      # (b * n, max_length)
        non_tensor_batch["multi_modal_inputs"] = model_inputs

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)

        # # prompt: left pad + response: right pad
        # # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        # response_position_ids = position_ids[..., -1:] + delta_position_id
        response_position_ids = torch.stack(response_position_ids, dim=0).to(input_ids.device)                    # (b*n, 3, max_length)
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.pad_2d_list_to_length(response_mask, 0, max_length=self.config.response_length).to(input_ids.device)                      # (b * n, max_length)
        multi_turn_mask = VF.pad_2d_list_to_length(multi_turn_mask, 0, max_length=self.config.response_length).to(input_ids.device)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        valid_lengths = torch.sum(attention_mask, dim=1)  # 按行求和
        max_valid_length = torch.max(valid_lengths).cpu()   
        min_valid_length = torch.min(valid_lengths).cpu()
        avg_valid_length = torch.mean(valid_lengths.float()).cpu()
        image_pad_counts = torch.sum(sequence_ids == 151655, dim=1)  # 按行统计
        max_image_pads = torch.max(image_pad_counts).cpu()
        min_image_pads = torch.min(image_pad_counts).cpu()
        avg_image_pads = torch.mean(image_pad_counts.float()).cpu()

        print(f"Size of prompt_ids: {input_ids.size()}")
        print(f"Size of response_ids: {response_ids.size()}")
        print(f"Size of sequence_ids: {sequence_ids.size()}")
        print(f"Valid Length - Max: {max_valid_length}, Min: {min_valid_length}, Avg: {avg_valid_length:.2f}")
        print(f"Image_pad Number - Max: {max_image_pads}, Min: {min_image_pads}, Avg: {avg_image_pads:.2f}")

        batch = TensorDict(
            {
                "prompts": input_ids,                   # origin prompt ids
                "responses": response_ids,              # new response ids
                "input_ids": sequence_ids,              # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,                             # new response mask
                "position_ids": position_ids,
                "multi_turn_mask": multi_turn_mask,
                # query-aware
                # 'image_to_query_mapping': image_to_query_mapping,
                # 'query_ids': query_ids,
                # 'query_attention_mask': query_mask
            },
            batch_size=batch_size,
        )
        for key, value in non_tensor_batch.items():
            if isinstance(value, np.ndarray) is False:
                non_tensor_batch[key] = np.array(value, dtype=object)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


