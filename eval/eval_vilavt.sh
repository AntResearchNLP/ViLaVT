#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=5,7 # 0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO

NPROC_PER_NODE=2
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}                     # 查看ip: ip -4 addr show eth0 	
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES-1}
NODE_RANK=${NODE_RANK:-0}                 # 当前节点编号：0 ~ NNODES-1


DATADIR="./benchmark"
script_paths=(
"vsibench_32frame"
"spar_bench"
"mmsi_bench"
"viewspatial_bench"
"ERQA"
# ============= vlmeval =============
"HRBench4K"
"HRBench8K"
# ============= single view frame spatial =============
"EmbSpatial"
"SpatialEval_spatialreal"
)

model_path=/path/to/checkpoint
model_name="ViLaVT"
output_dir="outputs"



# 遍历每个 dataset 并启动 torchrun
for dataset in "${script_paths[@]}"; do
    echo "🚀 Starting evaluation on dataset: $dataset"

    # 获取对应 image_folder
    image_folder=$DATADIR

    # 输入文件路径
    input_file="$DATADIR/${dataset}.json"
    if [ ! -f "$input_file" ]; then
        echo "❌ Input file not found: $input_file"
        continue
    fi

    # 启动单机 8 卡推理
    echo "Running in multi-node mode: nnodes=${NNODES}, node_rank=${NODE_RANK}, master_addr=${MASTER_ADDR}:${MASTER_PORT}"
    torchrun \
        --nproc_per_node=${NPROC_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        eval/eval_vilavt.py \
        --model-path "$model_path" \
        --model-name "$model_name" \
        --dataset "$dataset" \
        --input-file "$input_file" \
        --image-folder "$image_folder" \
        --output-dir "$output_dir" \
        --save-intermediate 

    echo "✅ Finished evaluation on $dataset"
done

echo "🎉 All evaluations completed!"