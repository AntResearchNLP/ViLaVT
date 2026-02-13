set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 


export CUDA_VISIBLE_DEVICES=2,3 # 0,1,2,3  4,5,6,7  8,9,10,11  12,13,14,15
split=0
all=1
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


model_path="/path/to/checkpoint"
model_name="ViLaVT"
MAX_PIXELS=$((8192*28*28))
echo "Processing shard $split of $all"


for ((i=0; i<${#script_paths[@]}; i++)); do
    dataset=${script_paths[i]}
    IMAGE_FOLDER=$DATADIR
    echo "${dataset}--${IMAGE_FOLDER}"

    if [ -z "$IMAGE_FOLDER" ]; then
        echo "Warning: No image folder defined for $dataset. Skipping..."
        continue
    fi
    RESULTDIR=./outputs_vllm/${model_name}/   
    mkdir -p $RESULTDIR

    # 输入文件路径
    input_file="$DATADIR/${dataset}.json"

    python vllm_eval/infer_vilavt_vllm.py \
        --model-path $model_path \
        --model-name ${model_name} \
        --data-type ${script_paths[i]} \
        --dataset $dataset \
        --input-file $input_file \
        --image-folder $IMAGE_FOLDER \
        --output-dir $RESULTDIR \
        --temperature 0.75 \
        --max-frames 32 \
        --max-pixels ${MAX_PIXELS} \
        --split $split \
        --all $all \
        --over_write 1
done