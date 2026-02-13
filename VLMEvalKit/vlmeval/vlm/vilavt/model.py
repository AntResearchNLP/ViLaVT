from __future__ import annotations

import copy
import logging
import os
import re
import warnings
import sys
import time

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoProcessor, AutoTokenizer

from ..base import BaseModel
from .prompt import ViLaVTPromptMixin
from .utils import (
    REASONING_SYS_PROMPT,
    IMAGE_INDEX_PROMPT_V2,
    RESPONSE_PROMPT,
    RESPONSE_PROMPT_FINAL,
    SIMPLE_SYS_PROMPT,
    generate_prompt_qa,
    generate_prompt_simple_qa,
    parse_output,
    zoomin,
    fetch_image
)


def ensure_image_url(image: str) -> str:
    if os.path.exists(image):
        return image
    raise ValueError(f"Invalid image: {image}")


def ensure_video_url(video: str) -> str:
    if os.path.exists(video):
        return video
    raise ValueError(f"Invalid video: {video}")


class ViLaVT(ViLaVTPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        # rounds of intermediate steps
        max_iterations=5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = "You are a helpful assistant.",
        post_process: bool = True,
        # if True, will try to only extract stuff wrapped in <answer> &
        # </answer>.
        verbose: bool = True,
        **kwargs,
    ):
        
        # 添加目标模块所在的目录到 sys.path
        module_path = 'ViLaVT/qwen_vl_finetune/vilavt'
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        # 现在可以正常导入
        # from modeling_vilavt_v3 import Qwen2_5_VLForConditionalGeneration
        from modeling_vilavt_v4_2_torch2_8 import Qwen2_5_VLForConditionalGeneration
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        embedding_model_path = "Qwen/Qwen3-Embedding-0.6B"        # hardcode
        self.text_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.model.eval()

        torch.cuda.empty_cache()

    def _extract_image_path(self, contents: list[dict[str, str]]):
        user_image_path_list = []
        content_history = copy.deepcopy(contents)
        for rou in content_history:
            if rou["type"] != "image":
                continue
            user_image_path_list.append(rou["value"])
        return user_image_path_list

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        user_image_path_list = self._extract_image_path(inputs)
        print(f"user_image_path: {user_image_path_list}")
        content = []
        image_idx = 0
        width, height = None, None
        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 28 * 28
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                img = fetch_image(item)
                (width, height) = img.size if img is not None else (None, None)
            elif s["type"] == "video":
                item = {
                    "type": "video",
                    "video": ensure_video_url(s["value"]),
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    import cv2

                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = (
                            frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR)
                        print(f"use {new_frame_count} for {s['value']}")
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe
            elif s["type"] == "text":
                item = {"type": "text",
                "text": generate_prompt_qa(
                            s['value'], 
                            user_image_path_list,
                            max_pixels=self.max_pixels,
                            min_pixels=self.min_pixels)}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
            if s["type"] == 'image':
                index_prompt = IMAGE_INDEX_PROMPT_V2.format(current_image_idx=image_idx, width=width, height=height)
                item = {"type": "text", "text": index_prompt}
                content.append(item)
                image_idx = image_idx + 1

        return content

    def _prepare_content_simple(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 28 * 28
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
            elif s["type"] == "video":
                item = {
                    "type": "video",
                    "video": ensure_video_url(s["value"]),
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    import cv2

                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = (
                            frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR)
                        print(f"use {new_frame_count} for {s['value']}")
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe
            elif s["type"] == "text":
                item = {
                    "type": "text",
                    "text": generate_prompt_simple_qa(
                        s["value"])}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _extract_box_answer(self, response):
        resp = response.split("\\boxed{")[-1]
        lt = len(resp)
        counter, end = 1, None
        for i in range(lt):
            if resp[i] == "{":
                counter += 1
            elif resp[i] == "}":
                counter -= 1
            if counter == 0:
                end = i
                break
            elif i == lt - 1:
                end = lt
                break
        if end is not None:
            response = resp[:end]
        return response

    def generate_inner_transformers(
            self,
            message,
            dataset=None):
        
        total_start_time = time.time() 
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )  # noqa: E501
            raise err

        messages = []
        messages.append({"role": "system", "content": REASONING_SYS_PROMPT})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        images, videos = process_vision_info([messages])
        print("images: ", images)

        if self.verbose:
            print(f"\033[31m{messages}\033[0m")


        #  -------   store history image manipulation
        original_image_url_list = self._extract_image_path(message)
        region_position_list = [[0., 0., 1., 1.] for _ in range(len(original_image_url_list))]
        query_list = ["" for _ in range(len(original_image_url_list))]
        img_to_query_mapping = list(range(len(original_image_url_list)))       # -1: no mapping query
        current_img_idx = len(images)
        # breakpoint()

        #   -------   outer loop for max_iterations -------
        iterations = self.max_iterations
        has_valid_answer = False
        while (iterations > 0) and (not has_valid_answer):
            step_start_time = time.time()
            iteration_num = self.max_iterations - iterations + 1

            # For each generation, we initialize a KV-Cache to speed up inference.
            # kv_cache = DynamicCache()
            if self.verbose:
                print(
                    f"\033[32m\n--- Iteration {self.max_iterations - iterations + 1} ---\033[0m"
                )

            # Step 1: 准备输入
            prep_start = time.time()
            text = self.processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )

            # encode query
            inputs['image_to_query_mapping'] = torch.tensor(img_to_query_mapping,  dtype=torch.long)
            # have to use qwen3-embedding model tokenizer, instead of qwen2.5-vl model tokenizer
            encoded_inputs = self.text_tokenizer(
                query_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50
                )
            query_ids, query_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            is_all_pad = (query_ids == self.text_tokenizer.pad_token_id).all(dim=1)  # [B]
            query_mask[is_all_pad] = 0  # 强制全 0
            inputs['query_ids'] = query_ids
            inputs['query_attention_mask'] = query_mask
            inputs = inputs.to("cuda")

            prep_time = time.time() - prep_start

            # Step 2: 模型生成
            gen_start = time.time()
            generated_ids = self.model.generate(
                **inputs, **self.generate_kwargs, 
                # past_key_values=kv_cache
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_text_segment = out[0]
            gen_time = time.time() - gen_start

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text_segment}]
            })

            
            # Step 3: 解析和处理
            parse_start = time.time()
             # Case 1: directly give answer
            parsed_output = None
            if "</answer>" in generated_text_segment:
                zoomin_time = 0
                parse_time = time.time() - parse_start
                pass
            else:
                content_feedback = []
                parsed_output = parse_output(generated_text_segment)
                print(f"Parsed Output: {parsed_output}")

                # zoomin images have been resized to multiples of 28 
                zoomin_start = time.time()
                zoomin_image_list, zoomin_region_position, zoomin_image_url_list, zoomin_query, valid_flag = zoomin(
                    images,
                    region_position_list,
                    original_image_url_list,
                    parsed_output,
                    max_pixels=self.max_pixels
                )
                zoomin_time = time.time() - zoomin_start

                if not valid_flag:
                    zoomin_feedback = zoomin_query
                    content_feedback.append({
                        "type": "text", 
                        "text": "<tool_response>" + f"[Error] Zoom-in failed: {zoomin_feedback}\n"
                    })
                    content_feedback.append({"type": "text", "text": RESPONSE_PROMPT_FINAL + "</tool_response>"})
                else:

                    # append to conversations
                    images.extend(zoomin_image_list)
                    region_position_list.extend(zoomin_region_position)
                    original_image_url_list.extend(zoomin_image_url_list)
                    query_list.append(zoomin_query)
                    img_to_query_mapping.extend([len(query_list)-1 for _ in range(len(zoomin_image_list))])        # new zoomin image mapping to zoomin query

                    
                    content_feedback.append({"type": "text", "text": "<tool_response>"})
                    for zoomin_image, region_position in zip(zoomin_image_list, zoomin_region_position):
                        zoomin_width, zoomin_height = zoomin_image.size
                        content_feedback.append({"type": "image"})
                        content_feedback.append({"type": "text", "text": IMAGE_INDEX_PROMPT_V2.format(current_image_idx=current_img_idx, width=zoomin_width, height=zoomin_height)})
                        current_img_idx += 1
                    if iterations -1 <= 1:
                        # last iteration have to give final response
                        content_feedback.append({"type": "text", "text": RESPONSE_PROMPT_FINAL + "</tool_response>"})
                    else:
                        content_feedback.append({"type": "text", "text": RESPONSE_PROMPT + "</tool_response>"})

                messages.append(
                    {"role": "user", "content": content_feedback})
                print(f"Feedback: {content_feedback}")
                parse_time = time.time() - parse_start

            # 打印当前step的时间统计
            step_total_time = time.time() - step_start_time
            print(f"\033[33m=== Step {iteration_num} Time Statistics ===")
            print(f"  Input Preparation: {prep_time:.3f}s")
            print(f"  Model Generation: {gen_time:.3f}s")
            print(f"  Zoom-in Processing: {zoomin_time:.3f}s")
            print(f"  Parsing & Feedback: {parse_time:.3f}s")
            print(f"  Step Total: {step_total_time:.3f}s")
            print(f"================================\033[0m")
            # --- Check for final answer tag in this segment ---
            if "</answer>" in generated_text_segment:
                has_valid_answer = True
                print("\033[32m--- Final answer tag found. ---\033[0m")
                if self.verbose:
                    print(
                        f"\033[32m\n--- End of processing (max iterations: {self.max_iterations},"
                        f"actual: {self.max_iterations - iterations + 1}) ---\033[0m"
                    )
                break
            else:
                if self.verbose:
                    print(
                            f"\033[31m--- Continue Reasoning ---\n"
                            f"{generated_text_segment}\n"
                            f",-------------------------\033[0m"
                    )
                    # print(f"\033[31m--- Continue reasoning. Query: {parsed_output} ---\033[0m")
            iterations -= 1
            
        if not has_valid_answer:
            # psycho: reinforce the model to answer
            print(
                f"\033[32m\n --- Fail to find a valid answer after {self.max_iterations} iterations."
                f"Reinforce to give final answer.---\033[0m"
            )

            messages = []
            if self.system_prompt is not None:
                messages.append(
                    {"role": "system", "content": SIMPLE_SYS_PROMPT})
            messages.append(
                {
                    "role": "user",
                    "content": self._prepare_content_simple(message, dataset=dataset),
                }
            )
            text = self.processor.apply_chat_template(
                [messages], tokenize=False, add_generation_prompt=True)

            original_image_url_list = self._extract_image_path(message)
            query_list = ["" for _ in range(len(original_image_url_list))]
            img_to_query_mapping = list(range(len(original_image_url_list))) 
            images, videos = process_vision_info([messages])
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
            # encode query
            inputs['image_to_query_mapping'] = torch.tensor(img_to_query_mapping,  dtype=torch.long)
            # have to use qwen3-embedding model tokenizer, instead of qwen2.5-vl model tokenizer
            encoded_inputs = self.text_tokenizer(
                query_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50
                )
            inputs['query_ids'] = encoded_inputs.input_ids
            inputs['query_attention_mask'] = encoded_inputs.attention_mask
            inputs = inputs.to("cuda")

            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_text_segment = out[0]
            

            # to align with the following processing procedure. wrap a <answer>
            # bracket.
            answer_match = re.search(
                r"<answer>(.*?)</answer>", generated_text_segment, re.DOTALL
            )
            if not answer_match:
                generated_text_segment = (
                    "<answer>" + generated_text_segment + "</answer>"
                )
            messages.append({           # content: str/list type
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text_segment}]
            })

        final_assistant_response = ""
        for msg in reversed(messages):
            if msg["role"] != "assistant":
                continue
            current_content_str = ""
            for item in msg["content"]:
                if item["type"] == "text":
                    current_content_str += item["text"]
            # Get the last full response from assistant
            final_assistant_response = current_content_str
            break
        # breakpoint()
        if self.post_process:
            print(
                f"\033[31m--- Final response ---\n{final_assistant_response}\n-------------------------\033[0m"
            )
            # Extract content within <answer> tags from the final assistant response
            answer_match = re.search(
                r"<answer>(.*?)</answer>", final_assistant_response, re.DOTALL
            )
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                final_answer = "No answer tag found in the final output."

            # Sometimes the answer is still wrapped in \boxed{}, keeping the behaviour of Qwen2.5-VL.
            # We extract the answer within this.
            match = re.search(r"\\boxed\{(.*?)\}", final_answer)
            if match:
                final_answer = self._extract_box_answer(final_answer)


            # 打印总时间
            total_time = time.time() - total_start_time
            print(f"\033[36m{'='*50}")
            print(f"TOTAL EXECUTION TIME: {total_time:.3f}s")
            print(f"{'='*50}\033[0m")
            if self.verbose:
                print(f"\033[32m{final_answer}\033[0m")
            return final_answer
        else:
            total_time = time.time() - total_start_time
            print(f"\033[36mTOTAL EXECUTION TIME: {total_time:.3f}s\033[0m")
            return final_assistant_response

    def generate_inner(self, message, dataset=None):

        return self.generate_inner_transformers(message, dataset=dataset)
