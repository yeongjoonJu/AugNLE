#!/bin/bash
# --captioning_anno_paths \
# "../captioning_data/nocaps/annotations/filtered_80.46_k-100_p-0.5_t-0.7_n_1.json" \
# --captioning_image_dirs \
# "../captioning_data/nocaps/image" \
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0 python trainer.py \
--mode train \
--cached_dir cached/nlx_gpt_base \
--experiment_name nlx_gpt_base_b32 \
--max_epochs 30 \
--ngpu 1 \
--warmup_steps 0 \
--checkpoints_dir ${CKPT_DIR} \
--img_size 224 \
--weight_decay 0.0 \
--nle_train_anno_path "../nle_anno/VQA-X/vqaX_train.json" \
--nle_valid_anno_path "../nle_anno/VQA-X/vqaX_val.json" \
--nle_train_image_dir "../images/train2014" \
--nle_valid_image_dir "../images/val2014" \
--train_batch_size 32 \
--learning_rate 2e-5 \
--gradient_accumulation_steps 1 \
--val_check_interval 1.0 \
--max_iter 1