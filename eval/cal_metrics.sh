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

# MODEL_NAME=ViLaVT-v0.4.2
# MODEL_NAME=ViLaVT-v0.4.3-ckpt_2100
# MODEL_NAME=vilavt_baseline_sr_spar
# MODEL_NAME="ViLaVT-baseline_rl_700"
# MODEL_NAME="ViLaVT-baseline_rl_1000"
MODEL_NAME="ViLaVT-v0.4.3.5-RL-step1250"
# MODEL_NAME=Thyme-RL
echo START Score Answer...

# OUTPUT_DIR=outputs
OUTPUT_DIR=outputs

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
        --result-file $RESULTDIR/${MODEL_NAME}_${DATASET}_results.jsonl \
        --output-result $RESULTDIR/${MODEL_NAME}_${DATASET}_scores.jsonl 
    echo "Evaluation complete for ${DATASET}"
    echo ""
done

echo "===========================================" 
echo "Evaluation Complete!"
echo "===========================================" 
