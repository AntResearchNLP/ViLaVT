# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2.5-VL model compatible with HuggingFace weights."""
from functools import cached_property, partial
from typing import (Callable, Iterable, List, Literal, Mapping, Optional, Set,
                    Tuple, TypedDict, Union, Any)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)

from vllm.config import VllmConfig, ModelConfig
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
# from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalDataDict,
                                MultiModalKwargs, MultiModalPlaceholderDict, 
                                MultiModalInputs, MultiModalFieldElem, MultiModalBatchedField,
                                MultiModalFieldConfig, MultiModalSharedField)
# from vllm.multimodal.hasher import MultiModalHashDict
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.model_executor.models.qwen2_vl import (Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo,
                       apply_rotary_pos_emb_vision)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model, maybe_prefix, 
                    make_empty_intermediate_tensors_factory,
                    merge_multimodal_embeddings)
from vllm.model_executor.models.vision import get_vit_attn_backend

# add by wjf
from vllm.model_executor.models.qwen3 import Qwen3Model
from transformers import Qwen3Config
from copy import deepcopy
from typing_extensions import NotRequired
from typing import Sequence

from transformers import AutoModel
# from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
# from .qwen2_vl import Qwen2VLDummyInputsBuilder as Qwen2_5_VLDummyInputsBuilder
# from .qwen2_vl import (Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo,
#                        apply_rotary_pos_emb_vision)
# from .utils import (AutoWeightsLoader, WeightsMapper,
#                     init_vllm_registered_model, maybe_prefix,
#                     merge_multimodal_embeddings)
# from .vision import get_vit_attn_backend


logger = init_logger(__name__)

# === Vision Inputs === #


class Qwen2_5_VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2_5_VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """

class Vilavt_ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """

    query_ids: NotRequired[torch.Tensor]
    """Shape: `(num_queries, L)`
    """
    query_attention_mask: NotRequired[torch.Tensor]
    """Shape: `(num_queries, L)`
    """
    image_to_query_mapping: NotRequired[torch.Tensor]
    """Shape: `(num_images)`
    """




class ViLavt_ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """
    query_ids: torch.Tensor
    """Shape: `(num_queries, L)`
    """
    query_attention_mask: torch.Tensor
    """Shape: `(num_queries, L)`
    """
    image_to_query_mapping: torch.Tensor
    """Shape: `(num_images)`
    """

# Qwen2_5_VLImageInputs = Union[Qwen2_5_VLImagePixelInputs,
#                               Qwen2_5_VLImageEmbeddingInputs]

Vilavt_ImageInputs = Union[Vilavt_ImagePixelInputs,
                              Qwen2_5_VLImageEmbeddingInputs]


class Qwen2_5_VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """

    second_per_grid_ts: torch.Tensor
    """
    The video time interval (in seconds) for each grid along the temporal 
    dimension in the 3D position IDs. Returned when `videos` is not `None`.
    """


class Qwen2_5_VLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all videos' features.
        Each tensor holds an video's features.
    - `torch.Tensor`: A tensor holding all videos' features
      (concatenation of all videos' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the videos.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2_5_VLVideoInputs = Union[Qwen2_5_VLVideoPixelInputs,
                              Qwen2_5_VLVideoEmbeddingInputs]

# === Vision Encoder === #


class Qwen2_5_VisionMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = False,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(in_features,
                                              hidden_features,
                                              bias=bias,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.gate_proj")
        self.up_proj = ColumnParallelLinear(in_features,
                                            hidden_features,
                                            bias=bias,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.up_proj")
        self.down_proj = RowParallelLinear(hidden_features,
                                           in_features,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        x_gate, _ = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up, _ = self.up_proj(x)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down


class Qwen2_5_VisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size)

        self.qkv = ColumnParallelLinear(input_size=embed_dim,
                                        output_size=3 * projection_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
        }:
            raise RuntimeError(
                f"Qwen2.5-VL does not support {self.attn_backend} backend now."
            )

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = tensor_model_parallel_all_gather(qkv)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition,
                     self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        if rotary_pos_emb is not None:
            use_flash_attn = self.attn_backend == _Backend.FLASH_ATTN
            q = apply_rotary_pos_emb_vision(q,
                                            rotary_pos_emb,
                                            use_flash_attn=use_flash_attn)
            k = apply_rotary_pos_emb_vision(k,
                                            rotary_pos_emb,
                                            use_flash_attn=use_flash_attn)

        if self.attn_backend == _Backend.FLASH_ATTN:
            # from vllm_flash_attn.flash_attn_interface import (
            #   flash_attn_varlen_func)
            from flash_attn import flash_attn_varlen_func

            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0,
                                            causal=False)

            context_layer = rearrange(output,
                                      "(b s) ... -> b s ...",
                                      b=batch_size)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            # Execute attention entry by entry for speed & less VRAM.
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                                 for x in [q_i, k_i, v_i])
                output_i = F.scaled_dot_product_attention(q_i,
                                                          k_i,
                                                          v_i,
                                                          dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None)

            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None)
        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Qwen2_5_VisionAttention_QA(Qwen2_5_VisionAttention):

    # def __init__(
    #     self,
    #     embed_dim: int,
    #     num_heads: int,
    #     projection_size: int,
    #     quant_config: Optional[QuantizationConfig] = None,
    #     use_text_instruction: bool = False,
    #     prefix: str = "",
    # ) -> None:
    #     super().__init__()
    #     # Per attention head and per partition values.
    #     self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
    #     self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
    #     self.hidden_size_per_attention_head = dist_utils.divide(
    #         projection_size, num_heads)
    #     self.num_attention_heads_per_partition = dist_utils.divide(
    #         num_heads, self.tp_size)

    #     self.qkv = ColumnParallelLinear(input_size=embed_dim,
    #                                     output_size=3 * projection_size,
    #                                     quant_config=quant_config,
    #                                     prefix=f"{prefix}.qkv")
    #     self.proj = RowParallelLinear(input_size=projection_size,
    #                                   output_size=embed_dim,
    #                                   quant_config=quant_config,
    #                                   prefix=f"{prefix}.proj")
        
    #     if use_text_instruction:
    #         self.instruct_key_proj = ColumnParallelLinear(input_size=embed_dim,
    #                                         output_size=projection_size,
    #                                         quant_config=quant_config,
    #                                         prefix=f"{prefix}.isntruct_k_proj")
    #         self.instruct_value_proj = ColumnParallelLinear(input_size=embed_dim,
    #                                         output_size=projection_size,
    #                                         quant_config=quant_config,
    #                                         prefix=f"{prefix}.isntruct_v_proj")

    #     # Detect attention implementation.
    #     self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
    #     if self.attn_backend not in {
    #             _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
    #     }:
    #         raise RuntimeError(
    #             f"Qwen2.5-VL does not support {self.attn_backend} backend now."
    #         )
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        use_text_instruction: bool = False,
        prefix: str = "",
    ) -> None:
        # 步骤1: 正确调用父类构造函数，让它完成所有共享的初始化工作
        super().__init__(embed_dim, num_heads, projection_size, 
                         quant_config, prefix)

        # 步骤2: 只初始化子类特有的属性
        self.use_text_instruction = use_text_instruction 
        if self.use_text_instruction:
            self.instruct_key_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=projection_size,
                quant_config=quant_config,
                prefix=f"{prefix}.isntruct_k_proj")
            self.instruct_value_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=projection_size,
                quant_config=quant_config,
                prefix=f"{prefix}.isntruct_v_proj")


    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        instruct_embeds: Optional[torch.Tensor] = None,
        instruct_masks: Optional[torch.Tensor] = None,
        instruct_rotary_pos_emb: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        layer_num=None,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        if not self.use_text_instruction or instruct_embeds is None:
            return super().forward(x, cu_seqlens, rotary_pos_emb)
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]
        num_heads = q.shape[2]
        

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))

        if rotary_pos_emb is not None:
            use_flash_attn = self.attn_backend == _Backend.FLASH_ATTN
            q = apply_rotary_pos_emb_vision(q,
                                            rotary_pos_emb,
                                            use_flash_attn=use_flash_attn)
            k = apply_rotary_pos_emb_vision(k,
                                            rotary_pos_emb,
                                            use_flash_attn=use_flash_attn)
        q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        assert self.attn_backend == _Backend.FLASH_ATTN, f"self.attn_backend: {self.attn_backend} != _Backend.FLASH_ATTN: {_Backend.FLASH_ATTN}"
        from flash_attn import flash_attn_varlen_func

        cu_seqlens_q = cu_seqlens
        # print(f"Layer-{layer_num}, cu_seqlens_q: {cu_seqlens_q}, cu_seqlens_k: {cu_seqlens_k}")
        if self.use_text_instruction and instruct_embeds is not None and instruct_masks is not None:
            B, T, D = instruct_embeds.shape
            instruct_k, _ = self.instruct_key_proj(instruct_embeds)  
            instruct_v, _ = self.instruct_value_proj(instruct_embeds)  
            # reshape: (B, T, head, head_dim)
            instruct_k = instruct_k.view(B, T, num_heads, -1)
            instruct_v = instruct_v.view(B, T, num_heads, -1)

            # 应用 RoPE（如果提供）
            if instruct_rotary_pos_emb is not None:
                use_flash_attn = self.attn_backend == _Backend.FLASH_ATTN
                # apply_rotary_pos_emb_vision need 
                # t: (batch_size, seqlen, nheads, headdim)
                # cos, sin: (seqlen_rotary, rotary_dim / 2)
                # To fit the shape, we merge all queries into one query in one batch
                instruct_k = instruct_k.view(1, B*T, num_heads, -1)  # (B*T, H, d)
                instruct_rotary_pos_emb = instruct_rotary_pos_emb.view(B*T, -1)  
                # print(f"instruct_k.shape: {instruct_k.shape}, instruct_rotary_pos_emb.shape: {instruct_rotary_pos_emb.shape}")
                instruct_k = apply_rotary_pos_emb_vision(instruct_k,
                                                instruct_rotary_pos_emb,
                                                use_flash_attn=use_flash_attn)  
                instruct_k = instruct_k.view(B, T, num_heads, -1)   # reshape to the correct shape

            valid_instruct_k = [instruct_k[b][instruct_masks[b].bool()] for b in range(B)]
            valid_instruct_v = [instruct_v[b][instruct_masks[b].bool()] for b in range(B)]

            # === 关键修复：使用 list + cat 构造 k/v，保持梯度 ===
            chunks_k = []
            chunks_v = []
            img_start = 0
            for i in range(len(cu_seqlens) - 1):
                q_len = cu_seqlens[i + 1] - cu_seqlens[i]
                # 图像部分 k/v
                img_k = k[img_start:img_start + q_len]
                img_v = v[img_start:img_start + q_len]
                chunks_k.append(img_k)
                chunks_v.append(img_v)
                img_start += q_len

                if i < len(valid_instruct_k) and len(valid_instruct_k[i]) > 0:   # 插入对应文本 instruction
                    chunks_k.append(valid_instruct_k[i])
                    chunks_v.append(valid_instruct_v[i])
            if chunks_k:
                k = torch.cat(chunks_k, dim=0)
            if chunks_v:
                v = torch.cat(chunks_v, dim=0)

            
            max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens_q,
                                            cu_seqlens_k=cu_seqlens_k,
                                            max_seqlen_q=max_seqlen_q,
                                            max_seqlen_k=max_seqlen_k,
                                            dropout_p=0,
                                            causal=False)

            context_layer = rearrange(output,
                                        "(b s) ... -> b s ...",
                                        b=batch_size)
        
        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output

class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_text_instruction: bool=False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.use_text_instruction = use_text_instruction
        self.attn = Qwen2_5_VisionAttention_QA(embed_dim=dim,
                                num_heads=num_heads,
                                projection_size=dim,
                                quant_config=quant_config,
                                use_text_instruction=use_text_instruction,
                                prefix=f"{prefix}.attn")

        self.mlp = Qwen2_5_VisionMLP(dim,
                                     mlp_hidden_dim,
                                     act_fn=act_fn,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.mlp")

    def forward(self, x: torch.Tensor, 
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                instruct_embeds: Optional[torch.Tensor] = None,
                instruct_masks: Optional[torch.Tensor] = None,
                instruct_rotary_pos_emb: Optional[torch.Tensor] = None,
                cu_seqlens_k: Optional[torch.Tensor] = None,
                layer_num=None,
        ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb=rotary_pos_emb,
                          instruct_embeds=instruct_embeds,
                          instruct_masks=instruct_masks,
                          instruct_rotary_pos_emb=instruct_rotary_pos_emb,
                          cu_seqlens_k=cu_seqlens_k,
                          layer_num=layer_num)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels,
                              hidden_size,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2"),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # args for get_window_index
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # self.blocks = nn.ModuleList([
        #     Qwen2_5_VisionBlock(
        #         dim=self.hidden_size,
        #         num_heads=self.num_heads,
        #         mlp_hidden_dim=vision_config.intermediate_size,
        #         act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
        #         norm_layer=norm_layer,
        #         quant_config=quant_config,
        #         prefix=f"{prefix}.blocks.{layer_idx}")
        #     for layer_idx in range(depth)
        # ])

        # wjf: modify attention block
        modules_list = []
        integrate_point = vision_config.integration_point
        depth = vision_config.depth
        for layer_idx in range(vision_config.depth):
            if vision_config.integration_point == 'all' or \
                integrate_point == 'early' and layer_idx < (depth // 2) or \
                integrate_point == 'late' and layer_idx >= (depth // 2) or \
                integrate_point == 'late2' and layer_idx >= (3 * depth // 4) or \
                integrate_point == 'late3' and layer_idx >= (depth // 4) or \
                integrate_point == 'sparse' and layer_idx in self.fullatt_block_indexes:
                use_text_instruction = True
               # print(f"QA Layer: Id-{layer_id}, {attn_implementation}")
            else:
                use_text_instruction = False
            layer = Qwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
                use_text_instruction = use_text_instruction
                )
            modules_list.append(layer)

        self.blocks = nn.ModuleList(modules_list)
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def rot_pos_emb_multi_image(self, grid_thw, image_to_query_mapping=None):
        pos_ids = []
        max_pos_offset = torch.tensor(0, dtype=grid_thw.dtype)  # wjf: 用于追踪当前最大位置编码，使其和hpos_ids，wpos_id在相同设备
        for idx, (t, h, w) in enumerate(grid_thw):  
            if idx == 0 or  image_to_query_mapping is None or (image_to_query_mapping[idx] != image_to_query_mapping[idx - 1]):
                max_pos_offset = torch.tensor(0, dtype=grid_thw.dtype)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)   # 对于同一样本对应的每张图片，位置编码从前一张图片的最大位置+1开始
            hpos_ids = hpos_ids + max_pos_offset
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1) + max_pos_offset
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            current_pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)            # 暂时：对于视频帧位置编码一致
            pos_ids.append(current_pos_ids)
            # 更新最大位置偏移，为下一张图片做准备
            max_pos_offset = torch.max(h, w).to(device=max_pos_offset.device, dtype=max_pos_offset.dtype) + max_pos_offset
        
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = torch.max(pos_ids) + 1          
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h,
                                                vit_merger_window_size,
                                                num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
                vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(
                0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
        instruct_embeds: Optional[torch.Tensor] = None,  # 新增：文本特征
        instruct_masks: Optional[torch.Tensor] = None,    # 新加：文本 attention mask
        image_to_query_mapping: Optional[torch.Tensor] = None      # 对应每张图片属于哪一个样本/instruct_embeds
    ) -> torch.Tensor:
        # patchify
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # compute position embedding
        # rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = self.rot_pos_emb_multi_image(grid_thw, image_to_query_mapping)     # wjf

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        cu_seqlens_q = [0]           # cu_seqlens为单图；cu_seqlens为多图。不同
        cu_seqlens_k = [0]
        query_rotary_pos_emb = None
        if instruct_embeds is not None and image_to_query_mapping is not None and instruct_masks is not None:
            query_to_image_mapping = {idx: [] for idx in range(max(image_to_query_mapping) + 1)}
            # 计算cu_seqlens_q和cu_seqlens_k
            for img_idx, query_idx in enumerate(image_to_query_mapping):
                query_idx = int(query_idx.item())
                query_to_image_mapping[query_idx].append(img_idx)

            max_pos_ids = []
            for query_idx, img_idx_list in query_to_image_mapping.items():
                offset = 0
                max_grid_size = 0           #  calculate the start pos_ids of queries in batch
                tmp_instruct_length = int(instruct_masks[query_idx].sum().item())
                for img_idx in img_idx_list:
                    # 统计同一样本的image_token范围: (old_offset, old_offset + t0*h0*w0 + t1*h1*w1 + ...)
                    num_img_token = grid_thw[img_idx, 0] * grid_thw[img_idx, 1] * grid_thw[img_idx, 2] 
                    offset = offset + int(num_img_token.item())
                    max_grid_size = max_grid_size + torch.max(grid_thw[img_idx, 1:]).item()
                
                cu_seqlens_q.append(cu_seqlens_q[-1] + offset)
                cu_seqlens_k.append(cu_seqlens_k[-1] + offset + tmp_instruct_length)
                max_pos_ids.append(max_grid_size)

            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=cu_seqlens.dtype).to(grid_thw.device)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=cu_seqlens.dtype).to(grid_thw.device)
            
            if image_to_query_mapping is not None:
                assert image_to_query_mapping.shape[0] == grid_thw.shape[0], \
                    f"image_to_query_mapping.shape[0]: {image_to_query_mapping.shape[0]} != grid_thw.shape[0]: {grid_thw.shape[0]}"        
            max_pos_ids = torch.tensor(max_pos_ids, device=grid_thw.device)
            base_query_pos_ids = torch.arange(instruct_embeds.shape[1], device=grid_thw.device)  # (T,)
            base_query_pos_ids = base_query_pos_ids.unsqueeze(0).expand(instruct_embeds.size(0), -1)  # 扩展为(num_samples, T)
            # print(f"Base Query Pos Ids: {base_query_pos_ids}")
            # print(f"max_pos_ids: {max_pos_ids}")
            query_pos_ids = base_query_pos_ids + max_pos_ids.unsqueeze(1)  # (num_samples, T)# 添加基于图像尺寸的偏移量
            query_pos_ids =  query_pos_ids.unsqueeze(-1).expand(-1,-1,2)                    # (num_samples, T, 2)
            max_query_pos = query_pos_ids.max() + 1
            query_rotary_pos_emb_full = self.rotary_pos_emb(max_query_pos)
            query_rotary_pos_emb = query_rotary_pos_emb_full[query_pos_ids].flatten(2)                  # (num_images, T, 40)

        # transformers
        hidden_states = hidden_states.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if blk.use_text_instruction:
                hidden_states = blk(hidden_states,
                                    cu_seqlens=cu_seqlens_q,
                                    rotary_pos_emb=rotary_pos_emb,
                                    instruct_embeds=instruct_embeds, 
                                    instruct_masks=instruct_masks, 
                                    instruct_rotary_pos_emb=query_rotary_pos_emb, 
                                    cu_seqlens_k=cu_seqlens_k,
                                    layer_num=layer_num)
            else:
                hidden_states = blk(hidden_states,
                                    cu_seqlens=cu_seqlens_now,
                                    rotary_pos_emb=rotary_pos_emb)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                #  for-else 结构：只有当 for 循环没有 break（即没有匹配任何分片规则）时才执行
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2_5_VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5_VLConfig)

    # v0.8.5.post1
    # def get_hf_processor(
    #     self,
    #     *,
    #     min_pixels: Optional[int] = None,
    #     max_pixels: Optional[int] = None,
    #     size: Optional[dict[str, int]] = None,
    #     fps: Optional[Union[float, List[float]]] = None,
    #     **kwargs: object,
    # ) -> Qwen2_5_VLProcessor:
    #     if fps is not None:
    #         kwargs["fps"] = fps

    #     return self.ctx.get_hf_processor(
    #         Qwen2_5_VLProcessor,
    #         image_processor=self.get_image_processor(min_pixels=min_pixels,
    #                                                  max_pixels=max_pixels,
    #                                                  size=size),
    #         **kwargs,
    #     )
    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )



# class VilavtMultiModalInputs(TypedDict):
#     """
#     Represents the outputs of
#     :class:`vllm.multimodal.processing.BaseMultiModalProcessor`,
#     ready to be passed to vLLM internals.
#     """

#     type: Literal["multimodal"]
#     """The type of inputs."""

#     prompt: str
#     """The processed prompt text."""

#     prompt_token_ids: list[int]
#     """The processed token IDs which includes placeholder tokens."""

#     token_type_ids: NotRequired[list[int]]
#     """The token type IDs of the prompt."""

#     mm_kwargs: MultiModalKwargs
#     """Keyword arguments to be directly passed to the model after batching."""

#     mm_hashes: Optional["MultiModalHashDict"]
#     """The hashes of the multi-modal data."""

#     mm_placeholders: MultiModalPlaceholderDict
#     """
#     For each modality, information about the placeholder tokens in
#     :code:`prompt_token_ids`.
#     """

#     # query-aware encoding fields
#     query_ids: NotRequired[Any]                 # this field can be missing and any type
#     query_attention_mask: NotRequired[Any]
#     image_to_query_mapping: NotRequired[Any]



class Qwen2_5_VLMultiModalProcessor(Qwen2VLMultiModalProcessor):

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputs:
        # print(f"In __call__\nPrompt: {prompt}")
        # print(f"mm_data: {mm_data}")
        return self.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # def _hash_mm_items(
    #     self,
    #     mm_items: MultiModalDataItems,
    #     hf_processor_mm_kwargs: Mapping[str, object],
    # ) -> dict[str, list[str]]:
    #     """Create MM hashes to be returned (only used in V1)."""

    #     # TODO: Use these hash keys for caching operations in apply_hf_processor
    #     # instead of rehashing.
    #     model_id = self.info.model_id

    #     return {
    #         modality: [
    #             MultiModalHasher.hash_kwargs(model_id=model_id,
    #                                          **{modality: item},
    #                                          **hf_processor_mm_kwargs)
    #             for item in items
    #         ]
    #         for modality, items in mm_items.items()
    #     }

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
        3. Extract information about the placeholder tokens from the
           processed token IDs.
        """
        # print(f"In apply()\nPrompt: {prompt}")
        # print(f"mm_data: {mm_data}")
        # pop query_ids and query_attention_mask
        query_ids = mm_data.get('query_ids')
        query_attention_mask = mm_data.get('query_attention_mask')
        image_to_query_mapping = mm_data.get('image_to_query_mapping')

        standard_mm_data = {
            k: v for k, v in mm_data.items()
            if k in {'image', 'video'}
        }
        mm_items = self._to_mm_items(standard_mm_data)
        
        mm_hashes = (self._hash_mm_items(mm_items, hf_processor_mm_kwargs)
                     if return_mm_hashes else None)
        # mm_hashes = None
        (
            prompt_ids,
            mm_kwargs,
            is_update_applied,
        ) = self._cached_apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
        )

        prompt_ids, prompt, mm_placeholders = self._maybe_apply_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            is_update_applied=is_update_applied,
        )

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }
        
        # ======================= 新增/修改的核心逻辑 =======================
        # 将你的自定义张量添加到 mm_kwargs 字典中
        # 3. 【核心注入与重建逻辑】c
        all_items: list[MultiModalKwargsItem] = []
        for modality in mm_kwargs.modalities:
            all_items.extend(mm_kwargs.get_items(modality))
        
        # b. 注入你的新字段到 items 列表中
        if query_ids is not None and \
            query_attention_mask is not None and \
            image_to_query_mapping is not None:

            batch_size = len(all_items)
            # 实例化官方的 SharedField，并传入正确的 batch_size
            shared_field = MultiModalSharedField(batch_size=batch_size)
            
            for item in all_items:
                # 确保我们只修改图片的 item
                if item.modality == 'image':
                    # 将全局数据作为新字段添加到这个 item 中
                    item['query_ids'] = MultiModalFieldElem(
                        'image', 'query_ids', query_ids, shared_field)      # field: "BaseMultiModalField"
                    
                    item['query_attention_mask'] = MultiModalFieldElem(
                        'image', 'query_attention_mask', query_attention_mask, shared_field)
                    
                    item['image_to_query_mapping'] = MultiModalFieldElem(
                        'image', 'image_to_query_mapping', image_to_query_mapping, shared_field)


            # query_id_elem = MultiModalFieldElem('image', 'query_ids', query_ids, MultiModalBatchedField())
            # query_mask_elem = MultiModalFieldElem('image', 'query_attention_mask', query_attention_mask, MultiModalBatchedField())
            # mapping_elem = MultiModalFieldElem('image', 'image_to_query_mapping', image_to_query_mapping, MultiModalBatchedField())
            # all_items.append(
            #     MultiModalKwargsItem.from_elems([query_id_elem, query_mask_elem, mapping_elem])
            # )
        mm_kwargs = MultiModalKwargs.from_items(all_items)
        # =================================================================
        # if query_ids is not None:
        #      breakpoint()

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholder_ranges,
        )


    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # print(f"hf_inputs: {hf_inputs}")
        # print(f"hf_processor_mm_kwargs: {hf_processor_mm_kwargs}")
        # return dict(
        #     **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
        #     second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        # )

        # 1. 调用父类的方法获取基础配置
        base_config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        
        # 2. 为你的新字段添加配置
        #    假设 query_ids 等是每个图片/请求一个，那么它们是 "batched" 字段
        #    如果它们是所有图片共享的，可能需要不同的 Field 类型
        new_config = {
            "query_ids": MultiModalFieldConfig.batched("image"),
            "query_attention_mask": MultiModalFieldConfig.batched("image"),
            "image_to_query_mapping": MultiModalFieldConfig.batched("image"),
        }

        # 3. 合并配置并返回
        #    注意：要用 base_config 更新 new_config，以防 new_config 覆盖了 base_config 中已有的 "video" 等配置
        new_config.update(base_config)
        return new_config


class VilavtDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2VLProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)                  # currently, we don't take video inputs

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token

        return image_token * num_images
        # num_images = mm_counts.get("image", 0)
        # num_videos = mm_counts.get("video", 0)

        # hf_processor = self.info.get_hf_processor()
        # image_token: str = hf_processor.image_token
        # video_token: str = hf_processor.video_token

        # return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        dummy_mapping = [0] * num_images

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "query_ids": torch.tensor([[151635]]),
            "query_attention_mask": torch.tensor([[1]]),
            "image_to_query_mapping": torch.tensor(dummy_mapping, dtype=torch.long),
        }



@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=VilavtDummyInputsBuilder)
class Qwen2_5_VLForConditionalGeneration_Vilavt(nn.Module, SupportsMultiModal,
                                         SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
        # wjf：将 HF 中的 text_encoder 映射到当前模块
        "text_encoder.": "text_encoder.",
        "text_proj.": "text_proj.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        text_encoder_config = Qwen3Config(**config.text_encoder_config)

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        # ----------- plan 1 ---------------
        # # print(f"maybe_prefix: {maybe_prefix(prefix, 'text_encoder')}")
        # # self.text_encoder = init_vllm_registered_model(
        # #     vllm_config=vllm_config,
        # #     prefix=maybe_prefix(prefix, "text_encoder"),
        # #     architectures=["Qwen3Model"],
        # #     hf_config=text_encoder_config 
        # # )
        # text_encoder_vllm_config = deepcopy(vllm_config)
        # text_encoder_vllm_config.model_config.hf_config = text_encoder_config
        # # print(f"text_encoder_vllm_config: {text_encoder_vllm_config}")
        # # print(f"text_encoder_vllm_config.cache_config: {text_encoder_vllm_config.cache_config}")
        # # print(f"text_encoder_vllm_config.kv_transfer_config: {text_encoder_vllm_config.kv_transfer_config}")
        # # print(f"type: {type(text_encoder_vllm_config.model_config.hf_config)}")
        # # print(f"text_encoder_vllm_config.model_config.hf_config: {text_encoder_vllm_config.model_config.hf_config}")
        # self.text_encoder = Qwen3Model(vllm_config=text_encoder_vllm_config, prefix=maybe_prefix(prefix, "text_encoder"))


        # ------------- plan 2 ---------------
        # 假设 Text Encoder 就是 Qwen3Model
        self.text_encoder = AutoModel.from_config(text_encoder_config)
        
        # 从 vLLM 管理的模型中加载权重到我们的独立 encoder 中
        # state_dict 的 key 可能需要做一些匹配和调整
        # vllm_model_state_dict = self.model.model.state_dict()
        # self.independent_text_encoder.load_state_dict(vllm_model_state_dict, strict=False)

        # print("text encoder: ", self.text_encoder)
        self.text_proj = ColumnParallelLinear(input_size=text_encoder_config.hidden_size,
                                              output_size=config.vision_config.hidden_size,
                                              bias=True,
                                              gather_output=True,  # If true, call all-gather on output and make Y available to all GPUs
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(prefix, ""))

        self.make_empty_intermediate_tensors = (self.language_model.make_empty_intermediate_tensors)

        # self.make_empty_intermediate_tensors = (
        #     make_empty_intermediate_tensors_factory(
        #         ["hidden_states", "residual"], config.hidden_size))

    # @cached_property
    # def sampler(self):
    #     if hasattr(self.language_model, "sampler"):
    #         return self.language_model.sampler

    #     return get_sampler()

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim} "
                                 f"(shape={mm_input.shape})")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Vilavt_ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        # ======================= 新增逻辑 =======================
        # 从 kwargs 中解析出你的新变量
        query_ids = kwargs.pop("query_ids", None)
        query_attention_mask = kwargs.pop("query_attention_mask", None)
        image_to_query_mapping = kwargs.pop("image_to_query_mapping", None)
        # ======================================================

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Vilavt_ImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw,
                                              query_ids=query_ids,
                                              query_attention_mask=query_attention_mask,
                                              image_to_query_mapping=image_to_query_mapping)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw)

    def _process_image_input(
            self,
            image_input: Vilavt_ImageInputs) -> tuple[torch.Tensor, ...]:

        # print(f"In Line-1145 in qwen2_5_vl_vilavt, _process_image_input: image_input = {image_input}")
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            query_ids = image_input.get("query_ids")
            query_attention_mask = image_input.get("query_attention_mask")
            image_to_query_mapping = image_input.get("image_to_query_mapping")
            if query_ids is not None and query_attention_mask  is not None and image_to_query_mapping is not None:
                text_outputs =self.text_encoder(query_ids, query_attention_mask)
                instruct_features = text_outputs.last_hidden_state 
                instruct_features, _ = self.text_proj(instruct_features)
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw, instruct_embeds=instruct_features, instruct_masks=query_attention_mask,  image_to_query_mapping=image_to_query_mapping)
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("pixel_values_videos",
                             "video_embeds") and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(
                    **kwargs)
        return modalities

    def get_multimodal_embeddings(
            self, **kwargs) -> Optional[tuple[torch.Tensor, ...]]:
        # print(f"kwargs in Line 1263, get_multimodal_embeddings(): {kwargs}")
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += video_embeddings
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[tuple[torch.Tensor, ...]] = None,
        video_input: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )

        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2.5-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
            second_per_grid_ts: Tensor `(num_videos)` of video time interval (
                in seconds) for each grid along the temporal dimension in the
                3D position IDs. `None` if no videos are passed.
        """
        
        # print(f"input_ids: {input_ids}")
        # print(f"positions: {positions}")
        # print(f"intermediate_tensors: {intermediate_tensors}")
        # print(f"input_embeds: {inputs_embeds}")
        # print(f"**kwargs: {kwargs}")

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        # sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[SamplerOutput]:
    #     return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.",
            tower_model="visual.merger.")


    @staticmethod
    def is_backend_compatible() -> bool:
        """
        Statically checks if the model is compatible with the vLLM backend.

        This is a 'pass-through' check. We claim compatibility because we know
        that the actual computation is delegated to a vLLM-compatible
        internal module (self.language_model).
        """
        return True
