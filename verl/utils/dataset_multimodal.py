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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from qwen_vl_utils import fetch_image

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import json


SYSTEM_PROMPT = """You are a helpful assistant. 
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

IMAGE_INDEX_PROMPT="""The index of the provided image is {index}.
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


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = "",
        max_pixels: int = 8192*28*28,
        min_pixels: int = 16*28*28,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = SYSTEM_PROMPT # system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        # if os.path.isdir(data_path):
        #     self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        # elif os.path.isfile(data_path):
        #     self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        # else:  # remote dataset
        #     self.dataset = load_dataset(data_path, split=data_split)
        self.dataset = [json.loads(x) for x in open(data_path, "r")]
        # self.dataset = HFDataset.from_list(self.dataset)

    def __len__(self):
        return len(self.dataset)

    
    # def process_image(self, image_path):
    #     """处理图像"""
    #     return fetch_image({"image": image_path, "max_pixels": self.max_pixels})


    def prepare_content(self, question, image_paths, min_pixels, max_pixels, image_size):
        content = []
        # 添加图片和索引提示
        for image_idx, path in enumerate(image_paths):
            # 图片项
            item = {"type": "image", "image": path}
            if min_pixels is not None:
                item["min_pixels"] = min_pixels
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)
            # 图片索引提示
            index_prompt = IMAGE_INDEX_PROMPT.format(index=image_idx)
            item = {"type": "text", "text": index_prompt}
            content.append(item)
        
        # 构建问题提示
        width, height = image_size
        
        if len(image_paths) > 1:
            image_instruction = VIDEO_QUESTION_PROMPT.format(
                n_frames=len(image_paths), 
                width=width, 
                height=height, 
                n_frames_1=len(image_paths) - 1
            )
        else:
            image_instruction = IMAGE_QUESTION_PROMPT.format(
                width=width, 
                height=height
            )
        
        prompt = USER_PROMPT.format(
            image_instruction=image_instruction, 
            question=question
        )
        
        # 添加问题文本（修正：添加 type 字段）
        item = {"type": "text", "text": prompt}
        content.append(item)
        
        return content


    def __getitem__(self, index):
        """获取数据项"""
        row_dict: dict = self.dataset[index].copy()
        max_pixels = row_dict.get('max_pixels', self.max_pixels)
        
        images = [fetch_image({"image": image_path, "max_pixels": max_pixels}) for image_path in row_dict[self.image_key] ]
        messages = []
        messages.append({
            "role": "system", 
            "content": self.system_prompt
        })
        messages.append({
            "role": "user", 
            "content": self.prepare_content(
                question=row_dict[self.prompt_key],
                image_paths=row_dict[self.image_key],
                min_pixels=self.min_pixels,
                max_pixels=max_pixels,
                image_size=images[0].size  # ← 修正参数名
            )
        })

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_data"] = {"image": images}
        row_dict["multi_modal_inputs"] = dict(model_inputs)             # num of <|image_pad|>: h*w //(merge_size*2)
        # row_dict.pop(self.image_key)
        
        # 处理position_ids
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["message"] = messages
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        
        return row_dict



from transformers import AutoTokenizer, AutoProcessor
def test_rlhf_dataset():
    # 1. 初始化必要的组件
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    
    # 2. 创建数据集实例
    dataset = RLHFDataset(
        data_path="ViLaVT/2_rl/rl_data/rl_data.jsonl",
        tokenizer=tokenizer,
        processor=processor,
        prompt_key = "question",
        answer_key = "answer",
        image_key = "image_path",
        max_prompt_length=10240,
        max_pixels=256*28*28,  # 1M pixels
        min_pixels=16384     # 64K pixels
    )
    
    # 3. 基本信息测试
    print(f"Dataset size: {len(dataset)}")
    
    # 定义要检查的关键字段
    essential_fields = [
        "input_ids", "attention_mask", "position_ids", 
        "multi_modal_data", "multi_modal_inputs",
        "ground_truth", "max_pixels", "max_num_steps", "image_path"
    ]
    
    # 记录错误样本
    error_samples = []
    
    # 遍历所有样本
    length = len(dataset)
    for idx in range(length):
        try:
            # 获取样本
            sample = dataset[idx]
            # print(sample)
            
            # 检查关键字段
            missing_fields = [field for field in essential_fields if field not in sample]
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")
            
            # 检查张量形状
            assert sample['input_ids'].shape == sample['attention_mask'].shape, \
                "input_ids and attention_mask shapes don't match"
            
            # 检查图像数据
            images = sample['multi_modal_data']['image']
            # if isinstance(images, list):
            #     for i, img in enumerate(images):
            #         assert hasattr(img, 'size'), f"Invalid image at index {i}"
            #         assert img.size[0] * img.size[1] <= 256*28*28, f"Image {i} too large: {img.size}"
            #         assert img.size[0] * img.size[1] >= 16384, f"Image {i} too small: {img.size}"
            # else:
            #     assert hasattr(images, 'size'), "Invalid image"
            #     assert images.size[0] * images.size[1] <= 256*28*28, f"Image too large: {images.size}"
            #     assert images.size[0] * images.size[1] >= 16384, f"Image too small: {images.size}"
            
            # 每100个样本打印一次进度和内存使用情况
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples")
                print(f"Current sample shapes:")
                print(f"- input_ids: {sample['input_ids'].shape}")
                print(f"- attention_mask: {sample['attention_mask'].shape}")
                print(f"- position_ids: {sample['position_ids'].shape}")
                print("---")
                breakpoint()
                
        except Exception as e:
            error_msg = f"Error in sample {idx}: {str(e)}"
            print(error_msg)
            error_samples.append((idx, error_msg))
            
    # 打印最终统计信息
    print("\nValidation completed!")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Successful samples: {len(dataset) - len(error_samples)}")
    print(f"Failed samples: {len(error_samples)}")
    
    # 如果有错误样本，打印详细信息
    if error_samples:
        print("\nError details:")
        for idx, error in error_samples:
            print(f"Sample {idx}: {error}")
    
    # 打印一个成功样本的详细信息
    if len(dataset) > 0 and len(error_samples) < len(dataset):
        # 找第一个成功的样本
        for idx in range(length):
            if idx not in [x[0] for x in error_samples]:
                sample = dataset[idx]
                print("\nExample of successful sample:")
                print(f"Data type: {sample['data_type']}")
                print(f"Input IDs shape: {sample['input_ids'].shape}")
                print(f"Attention mask shape: {sample['attention_mask'].shape}")
                print(f"Position IDs shape: {sample['position_ids'].shape}")
                print(f"Ground truth: {sample['ground_truth']}")
                
                images = sample['multi_modal_data']['image']
                if isinstance(images, list):
                    print(f"Number of images: {len(images)}")
                    for i, img in enumerate(images):
                        print(f"Image {i} size: {img.size}")
                else:
                    print(f"Image size: {images.size}")
                break

if __name__ == "__main__":
    test_rlhf_dataset()
