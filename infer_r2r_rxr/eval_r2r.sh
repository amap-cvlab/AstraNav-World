#!/bin/bash

CHUNKS=5
PREDICT_FUTURE_FRAMES=${PREDICT_FUTURE_FRAMES:-false}
FUTURE_ARGS=()
if [[ "$PREDICT_FUTURE_FRAMES" == true ]]; then
    : "${WAN_MODEL_PATH:?Set WAN_MODEL_PATH to the Diffusers-format Wan base}"
    FUTURE_ARGS=(--predict-future-frames --wan-model-path "$WAN_MODEL_PATH")
elif [[ "$PREDICT_FUTURE_FRAMES" != false ]]; then
    echo "PREDICT_FUTURE_FRAMES must be true or false" >&2; exit 2
fi
MODEL_PATH="./data/checkpoint-x"  #replace the checkpoint path here
GPUS=(0 2 3 5 6)

#R2R
CONFIG_PATH="./data/r2r_pano.yaml"
SAVE_PATH="./data/result_r2r"

for IDX in $(seq 0 $((CHUNKS-1))); do
    REAL_GPU=${GPUS[$(( IDX % ${#GPUS[@]} ))]}
    echo "Running chunk $IDX on GPU $REAL_GPU"
    sleep $(( IDX * 2 )) 
    CUDA_VISIBLE_DEVICES=$REAL_GPU nohup python -u run_infer.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --model-path $MODEL_PATH \
        --result-path $SAVE_PATH "${FUTURE_ARGS[@]}" &
done

wait
