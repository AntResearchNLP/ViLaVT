import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
import transformers
from transformers import AutoTokenizer
from qwen_vl_utils import fetch_image

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    # ----------- mask original code -----------
    default_system_message = "You are a helpful assistant."           

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        # ----------- mask original code -----------
        # try:
        #     if roles[source[0]["from"]] != roles["human"]:
        #         source = source[1:]
        # except:
        #     print(f'Key {source[0]["from"]} Error not in roles {roles}: . Sources: ', sources)            # modified

        input_id, target = [], []
        if source[0]['from'] != roles["system"]:
            input_id += tokenizer.apply_chat_template(
                [{"role": "system", "content": default_system_message}]
            )
            target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.embedding_model_tokenizer = AutoTokenizer.from_pretrained(data_args.embedding_model_path, model_max_length=data_args.max_query_length)
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        # default: min_pixels 3136, max_pixels 12845056
        # self.data_args.image_processor.max_pixels = data_args.max_pixels
        # self.data_args.image_processor.min_pixels = data_args.min_pixels
        # self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        # self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "images" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("images" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def process_image_unified_v2(self, image_file_list, max_pixels=None):
        if max_pixels is None:
            max_pixels = self.data_args.max_pixels
        processor = self.data_args.image_processor
        image_list = [
            fetch_image({"image": image, "max_pixels": max_pixels, "min_pixels": self.data_args.min_pixels})
            for image in image_file_list
        ]
        
        visual_processed = processor(image_list, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        grid_thw = visual_processed["image_grid_thw"]
        return image_tensor, grid_thw

    def process_image_unified_v3(self, image_file_list):
        image_list = [
            fetch_image({"image": image, "max_pixels": self.data_args.max_pixels, "min_pixels": self.data_args.min_pixels})
            for image in image_file_list
        ]
        image_list = []
        grid_thw = []
        for image in image_file_list:
            new_image = fetch_image({"image": image, "max_pixels": self.data_args.max_pixels, "min_pixels": self.data_args.min_pixels})
            width, height = new_image.size
            image_list.append(image)
            grid_thw.append([1,width//28, height//28])

        grid_thw = torch.tensor(grid_thw, dtype=torch.long)
        return image_list, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        # torchcodec_video = None
        # try:
        #     torchcodec_video = self.video_torchcodec(video_file)
        #     return torchcodec_video
        # except Exception as e:
            # print(f"torchcodec attempt failed: {e}")
        raise ValueError(f"Failed to process video: no valid video data obtained")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    # def video_torchcodec(self, video_file):
    #     device = "cpu"  # or e.g. "cuda"
    #     decoder = VideoDecoder(video_file, device=device)
    #     total_frames = decoder.metadata.num_frames
    #     avg_fps = decoder.metadata.average_fps
    #     video_length = total_frames / avg_fps
    #     interval = getattr(self.data_args, "base_interval", 4)

    #     num_frames_to_sample = round(video_length / interval)
    #     video_min_frames = getattr(self.data_args, "video_min_frames", 4)
    #     video_max_frames = getattr(self.data_args, "video_max_frames", 8)

    #     target_frames = min(
    #         max(num_frames_to_sample, video_min_frames), video_max_frames
    #     )
    #     frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    #     frame_idx = np.unique(frame_idx)
    #     frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
    #     video = frame_batch.data.cpu().numpy()
    #     return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        from time import time
        t1 = time()
        if "images" in sources[0]:
            t1_5 = time()
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["images"]
            max_pixels = self.list_data_dict[i].get('max_pixels', self.data_args.max_pixels)
            if isinstance(image_file, List):
                image_file = [os.path.join(image_folder, file) for file in image_file]
                # results = [self.process_image_unified(file) for file in image_file]     # v0
                # _, _ = self.process_image_unified_v3(image_file)         # v3
                image_tensor, grid_thw = self.process_image_unified_v2(image_file, max_pixels) # v2, most efficient
            else:
                image_file = [os.path.join(image_folder, image_file)]
                image_tensor, grid_thw = self.process_image_unified_v2(image_file)
                # image_list, grid_thw = self.process_image_unified_v3(image_file)
            process_image_time = time() - t1_5
            # print(f"Total process image time: {process_image_time}")
            # print(f"image tensor shape: {image_tensor.shape}")
            # print(f"grid_thw.shape: {grid_thw.shape}")
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw
            ]
        img_time = time() - t1

        t2 = time()
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        video_time = time() - t2
        t3 = time()
        # breakpoint()
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=grid_thw,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        tokenize_time = time() - t3
        t4 = time()

        if "images" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )
        # breakpoint()
        rope_time = time() - t4
        
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "images" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image_tensor
            data_dict["image_grid_thw"] = grid_thw
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        t5 = time()
        # ---------------------- process query-aware encoding ----------------------
        # only query-aware
        queries = sources[0].get('queries', None)
        assert queries is not None, f"quereis is None"
        assert len(queries) == 1, f"len(queries): {len(queries)} != 1" 
        queries = [f"Query: {queries[0]}", f"Query: {queries[0]}"]
        image_to_query_mapping = [0, 1]

        # 编码 queries
        encoded_inputs = self.embedding_model_tokenizer(
            queries,
            padding='max_length',                        # 'max_length': padding至最大长度 ; True: 自动 padding 到 batch 内最长序列
            truncation=True,
            max_length=self.data_args.max_query_length,
            return_tensors="pt",
        )
        query_ids, query_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
        # 对于空字符串，强制mask全为0
        is_all_pad = (query_ids == self.embedding_model_tokenizer.pad_token_id).all(dim=1)  # [B]
        query_mask[is_all_pad] = 0  # 强制全 0
        
        image_to_query_mapping = torch.tensor(image_to_query_mapping, dtype=torch.long)         # transform to tensor
        data_dict['image_to_query_mapping'] = image_to_query_mapping
        data_dict['query_ids'] = query_ids
        data_dict['query_attention_mask'] = query_mask

        query_time = time() - t5
        t6 = time()
        merge_time = time() - t6
        total_time = time() - t1
        # 📊 打印耗时分析（可用于调试）
        # print(f"[Profile] Sample {i}:")
        # print(f"  ├─ Image:  {img_time:.3f}s")
        # print(f"  ├─ Video:  {video_time:.3f}s")
        # print(f"  ├─ Tokenize: {tokenize_time:.3f}s")
        # print(f"  ├─ RoPE:   {rope_time:.3f}s")
        # print(f"  ├─ Query:  {query_time:.3f}s")
        # print(f"  ├─ Merge:  {merge_time:.3f}s")
        # print(f"  └─ Total:  {total_time:.3f}s")
        # breakpoint()
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    max_query_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        # -------------------- add new section  -------------------
        if "image_to_query_mapping" in instances[0]:
            query_count_offset = 0
            global_image_to_query_mapping = []

            for instance in instances:
                mapping = instance["image_to_query_mapping"]    # e.g., [0,1,1,2]
                num_queries = instance["query_ids"].shape[0]    # (N, L) 
                if query_count_offset > 0:
                    mapping = mapping + query_count_offset
                global_image_to_query_mapping.append(mapping)
                query_count_offset += num_queries               # accumulated offset

            image_to_query_mapping = torch.cat(global_image_to_query_mapping, dim=0)
            batch["image_to_query_mapping"] = image_to_query_mapping

        if "query_ids" in instances[0]:
            query_ids_list = [instance["query_ids"] for instance in instances]
            query_ids = torch.cat(query_ids_list, dim=0)        # (total_N, L) padding already
            query_ids = query_ids[:, : self.max_query_length]
            batch["query_ids"] = query_ids

        if "query_attention_mask" in instances[0]:
            mask_list = [instance["query_attention_mask"] for instance in instances]
            query_attention_mask = torch.cat(mask_list, dim=0)  # (total_N, L) padding already
            query_attention_mask = query_attention_mask[:, : self.max_query_length]
            batch["query_attention_mask"] = query_attention_mask
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        # ! to be done
        batch["queries"] = None
        batch["image_to_query_mapping"] = None
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_query_length=data_args.max_query_length)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    from train.argument import DataArguments
    config = {
        "dataset_use": "spar_pilot", # "sr_91k%5,spar7m%5,sr_91k_text%5,spar7m_text%5", #"sr_91k%10,spar7m%10",sr_91k_text%5,
        "video_max_frames": 8,
        "video_min_frames": 4,
        "data_flatten": False,
        "data_packing": False,
        "base_interval": 2,
        "max_pixels": 451584,
        "min_pixels": 3136,
        "video_max_frame_pixels": 25088,
        "video_min_frame_pixels": 3136,
        "joint_image": False,
        "max_query_length": 200
    }

    data_args: DataArguments = DataArguments(**config)
    data_args.image_processor = transformers.AutoProcessor.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct',
        ).image_processor
    data_args.model_type = "qwen2.5vl"
    data_args.embedding_model_path="Qwen/Qwen3-Embedding-0.6B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-VL-7B-Instruct',
        # model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )
    data_module = make_supervised_data_module(tokenizer, data_args)
    train_dataset = data_module['train_dataset']
    length_list = []
    num_samples = len(train_dataset)
    from tqdm import tqdm
    for i in tqdm(range(num_samples), desc='Counting Length',  unit='item'):
        tmp = train_dataset._get_item(i)
        length = tmp['input_ids'].shape[1]
        length_list.append(length)
        # breakpoint()
        if length > 24576:
            print(f"Idx-{i} length > 24576: {length}")
            
    

    avg_length = np.mean(length_list)
    max_length = np.max(length_list)
    min_length = np.min(length_list)
    # 打印结果
    print(f"Sampled {num_samples} examples:")
    print(f"  Min Length:  {min_length}")
    print(f"  Max Length:  {max_length}")
    print(f"  Avg Length:  {avg_length:.2f}")
    # breakpoint()
    pass
