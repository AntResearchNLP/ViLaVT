#!/bin/bash
script_paths=(
"vsibench_32frame"
"spar_bench"
"mmsi_bench"
"ERQA"
"HRBench4K"
"HRBench8K"
"SpatialEval_spatialreal"
"EmbSpatial"
)

# MODEL_NAME="ViLaVT-v0.4.3-ckpt_2100"
# MODEL_NAME=ViLaVT-v0.4.3-ckpt_2100-RL              # ViLaVT-v0.4.2
# MODEL_NAME=ViLaVT-v0.4.3.2-RL-step_50
# MODEL_NAME=vilavt_baseline_sr_spar
# MODEL_NAME="ViLaVT-v0.4.3.3-RL-step250"
# MODEL_NAME="ViLaVT-v0.4.3.4-RL-step550"
# MODEL_NAME="ViLaVT-v0.4.5.4-RL-step100"
# MODEL_NAME="ViLaVT-v0.4.3.5-RL-step250"
# MODEL_NAME="ViLaVT-v0.4.3.5-RL-step600"
# MODEL_NAME="ViLaVT-v0.4.3.5-RL-step1000"
MODEL_NAME="ViLaVT-v0.4.3.5-RL-step1250"
# MODEL_NAME="ViLaVT-v0.4.3.5-RL-step1450"
echo START Score Answer...

# OUTPUT_DIR=outputs
OUTPUT_DIR=outputs_vllm
seed=923                   # (0 42 123 1234 2024 2026 12345 3407 0826 0923) 

for ((i=0; i<${#script_paths[@]}; i++)); do
    DATASET=${script_paths[i]}

    RESULTDIR=./${OUTPUT_DIR}/${MODEL_NAME}
    echo ""
    echo "#----------------------------------------#"
    echo "Processing dataset: ${QUESTION_FILE}"
    echo "Results path: $RESULTDIR/${MODEL_NAME}_${DATASET}_results.jsonl" 
    echo "#----------------------------------------#"
    python eval/cal_metrics.py \
        --dataset $DATASET \
        --question-file $RESULTDIR/${MODEL_NAME}_${DATASET}_results.jsonl \
        --result-file $RESULTDIR/${MODEL_NAME}_${DATASET}_seed_${seed}_results.jsonl \
        --output-result $RESULTDIR/${MODEL_NAME}_${DATASET}_seed_${seed}_scores.jsonl 
    echo "Evaluation complete for ${DATASET}"
    echo ""
done

echo "===========================================" 
echo "Evaluation Complete!"
echo "===========================================" 
