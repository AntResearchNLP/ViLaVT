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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import sys
# ========== 在最开始添加 ==========
# 注册自定义模型
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

module_path = 'path/to/ViLaVT/qwen_vl_finetune/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from vilavt import (Qwen2_5_VLForConditionalGeneration_Vilavt, VilavtConfig)

# 注册
AutoConfig.register("vilavt", VilavtConfig)
AutoModelForCausalLM.register(VilavtConfig, Qwen2_5_VLForConditionalGeneration_Vilavt)
# ========== 添加结束 ==========

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import  GcotRewardManager     # CustomRewardManager,
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        config.deep_post_init()
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = GcotRewardManager(tokenizer=tokenizer, compute_score=config.worker.reward.compute_score)
        val_reward_fn = GcotRewardManager(tokenizer=tokenizer, compute_score=config.worker.reward.compute_score)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)
    ppo_config.data.system_prompt = """You are a helpful assistant. 
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
<tool> {"region": [{"index": 0, "bbox_2d": [100, 200, 300, 400]}], "query": "Look for the red button"} </tool>"""


    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
