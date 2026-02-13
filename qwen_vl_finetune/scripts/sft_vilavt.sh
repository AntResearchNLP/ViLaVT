#!/bin/bash

export TORCHINDUCTOR_COMPILE_THREADS=1  # 将编译线程数限制为1，防止并行编译风暴
export PYTHONFAULTHANDLER=1   # 崩溃时打印 traceback

# Distributed training configuration
NPROC_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}                     # 查看ip: ip -4 addr show eth0 	
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}                 # 当前节点编号：0 ~ NNODES-1

# DeepSpeed configuration
# deepspeed=./scripts/zero3.json
deepspeed=./scripts_vilavt/ds_z2_config.json          # ds_z2_config, ds_z3_config

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=5e-5
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_vilavt.py

# Dataset configuration (replace with public dataset names)
datasets=sr_91k_v2,spar7m_v2,vgr_v2,thyme_2turn_v2,sr_91k_text,spar7m_text,thyme_text_v2,vica_cot,vica_text

# Output configuration
run_name="vilavt-sft"
output_dir=./saves/vilavt_sft

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_ ${datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --tune_text_encoder False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epoch 3.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 200704 \
    --min_pixels 3136 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 4 \
    --learning_rate ${lr} \
    --mm_projector_lr ${lr} \
    --vision_tower_lr ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 24576 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --embedding_model_path Qwen/Qwen3-Embedding-0.6B \
    --query_aware_version intra_then_inter \
    --integration_point late \
    --max_query_length 50"

# Launch training
echo "Running in multi-node mode: nnodes=${NNODES}, node_rank=${NODE_RANK}, master_addr=${MASTER_ADDR}:${MASTER_PORT}"
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         ${entry_file} ${args}
