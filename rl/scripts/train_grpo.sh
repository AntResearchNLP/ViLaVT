#!/bin/bash
set -x

export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 
MODEL_PATH="path/to/vilavt_sft_modified"

PROJECT_NAME="ViLaVT"
EXPERIMENT_NAME="vilavt"

# The system_prompt defined here has no effect; it has already been implemented directly in the code.

SYSTEM_PROMPT=""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m verl.trainer.main \
    config=./rl/scripts/config.yaml \
    data.train_files=path/to/rl_data.jsonl \
    data.val_files=path/to/rl_data.jsonl \
    data.max_response_length=13312 \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.rollout_batch_size=64 \
    algorithm.kl_coef=0.0 \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    worker.actor.optim.lr=1.0e-6  \
    worker.actor.global_batch_size=32 \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.freeze_text_encoder=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.padding_free=true \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.55 \
    worker.rollout.n=4 \
    worker.rollout.num_llm_calls_available=10 \
    worker.rollout.limit_images=55 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path="checkpoints/rl/${EXPERIMENT_NAME}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}"