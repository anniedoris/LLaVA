#!/bin/bash

export TRITON_CACHE_DIR=/orcd/data/faez/001/annie/huggingface/triton
export HF_HOME=/orcd/data/faez/001/annie/huggingface

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /orcd/data/faez/001/annie/llava/finetune_data/cadquery_train_data_synthbal_4096.json \
    --image_folder /orcd/data/faez/001/annie/llava/finetune_data/synth_bal_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /orcd/data/faez/001/annie/llava/checkpoints/llava-v1.5-13b-pretrain-4gpu-4096/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /orcd/data/faez/001/annie/llava/checkpoints/llava-v1.5-13b-4gpu-cadquery-4096-synthbal \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard