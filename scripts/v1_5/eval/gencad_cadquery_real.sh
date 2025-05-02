#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b-4gpu-cadquery-4096-5xmultiview"
SPLIT="cadquery_test_data_ten_real_red_orientation"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /orcd/data/faez/001/annie/llava/checkpoints/$CKPT \
        --question-file /orcd/data/faez/001/annie/llava/eval/gencad_real/$SPLIT.jsonl \
        --image-folder /orcd/data/faez/001/annie/llava/eval/ten_real_red_orientation \
        --answers-file /orcd/data/faez/001/annie/llava/checkpoints/$CKPT/eval/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --max_new_tokens 3450 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/orcd/data/faez/001/annie/llava/checkpoints/$CKPT/eval/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /orcd/data/faez/001/annie/llava/checkpoints/$CKPT/eval/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

