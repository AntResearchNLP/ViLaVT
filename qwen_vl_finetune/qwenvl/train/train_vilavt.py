4# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import subprocess

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class, print_trainable_parameters, print_trainable_parameters_visual


from transformers import (
    Qwen2VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
)

# from qwenvl.data.data_vilavt import make_supervised_data_module
# from qwenvl.data.data_vilavt_v2 import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoModel, AutoConfig

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        # DeepSpeed 下也需要手动保存 tokenizer
        if trainer.tokenizer is not None and trainer.args.should_save:
            trainer.tokenizer.save_pretrained(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def set_model(model_args, model):

    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    
    if model_args.tune_text_encoder:
        for n, p in model.text_encoder.named_parameters():
            p.requires_grad = True
        for n, p in model.text_proj.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.text_encoder.named_parameters():
            p.requires_grad = False
        for n, p in model.text_proj.named_parameters():
            p.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "qa_mi" in training_args.output_dir:
        print("Query Aware and Multi Image")
        from qwenvl.data.data_vilavt_qa_mi import make_supervised_data_module
    elif "multi_image" in training_args.output_dir:
        print("Only Multi-Image")
        from qwenvl.data.data_vilavt_multi_image import make_supervised_data_module
    elif "query_aware" in training_args.output_dir:
        print("Only Query-Aware")
        from qwenvl.data.data_vilavt_qa import make_supervised_data_module
    else:
        from qwenvl.data.data_vilavt_v2 import make_supervised_data_module

    
    if model_args.query_aware_version in ['late_fusion', 'late_fusion-multi_image', 'intra_then_inter']:
        from vilavt.modeling_vilavt_v4_2_torch2_8 import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2_5_VLModel,
        )
        print("Use 'Qwen2_5_VLForConditionalGeneration from modeling_vilavt_v3")
    else:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

    Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = print_trainable_parameters_visual
    Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # if "qwen2.5" in model_args.model_name_or_path.lower():
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.query_aware_version in ['late_fusion', 'late_fusion-multi_image'] and model_args.integration_point != "":
        config.vision_config.integration_point = model_args.integration_point
        if model_args.integration_point == "late":
            config.vision_config.fullatt_block_indexes= list(range(16, 32))
        elif model_args.integration_point == "late2":
            config.vision_config.fullatt_block_indexes= list(range(24, 32))

    elif model_args.query_aware_version in ['intra_then_inter']:
        config.vision_config.integration_point = model_args.integration_point
        # 其实可以不用加，优先使用qa层
        if model_args.integration_point == "late":
            config.vision_config.fullatt_block_indexes= [7,15]
        elif model_args.integration_point == "late2":
            config.vision_config.fullatt_block_indexes= [7,15,23]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    # init embedding model
    target_dtype = model.dtype
    embedding_model_path = model_args.embedding_model_path
    model.text_encoder = AutoModel.from_pretrained(
            embedding_model_path, 
            torch_dtype=target_dtype,       # ✅ 显式指定精度
            trust_remote_code=True).to(model.device)
    print(f"model.dtype: {model.dtype}, text_encoder.dtype: {model.text_encoder.dtype}, model.text_proj dtype: {model.text_proj.weight.dtype}")
    data_args.embedding_model_path = embedding_model_path
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False
    

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)
    
    if torch.distributed.get_rank() == 0:
        all_requires_grad = all(param.requires_grad for param in model.text_encoder.parameters())
        # 在qwenvl.train.trainer中重载
        print("Data Args: ", data_args)
        print("Model Args: ", model_args)
    
        print(f"All text encoder parameters require gradient: {all_requires_grad}")
        print(f"Qwen Config: {config}")
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # if torch.distributed.get_rank() == 0:
    #     breakpoint()
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if torch.distributed.get_rank() == 0:
        print(f"save model to path: {training_args.output_dir}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")              # must use flash_attention_2