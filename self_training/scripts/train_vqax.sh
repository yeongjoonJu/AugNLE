#!/bin/bash
# --captioning_anno_paths \
# "../captioning_data/nocaps/annotations/filtered_80.46_k-100_p-0.5_t-0.7_n_1.json" \
# --captioning_image_dirs \
# "../captioning_data/nocaps/image" \
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0 python trainer.py \
--mode train \
--cached_dir cached/self_e_nocap_flickr \
--experiment_name flickr_nocap_b64_e5_lr5e-4 \
--max_epochs 5 \
--ngpu 1 \
--warmup_ratio 0.1 \
--img_encoded \
--checkpoints_dir ${CKPT_DIR} \
--img_size 224 \
--weight_decay 0.0 \
--nle_train_anno_path "../nle_anno/VQA-X/vqaX_train.json" \
--nle_valid_anno_path "../nle_anno/VQA-X/vqaX_val.json" \
--nle_train_image_dir "../images/train2014" \
--nle_valid_image_dir "../images/val2014" \
--captioning_anno_paths \
../captioning_data/nocaps/annotations/filtered_75.09_k-100_p-0.7_t-1.0_n_1.json \
../captioning_data/flickr30k/annotations/filtered_72.22_k-100_p-0.7_t-1.0_n_1.json \
--captioning_image_dirs \
../captioning_data/nocaps/image \
../captioning_data/flickr30k/image \
--train_batch_size 64 \
--eval_batch_size 16 \
--learning_rate 5e-4 \
--gradient_accumulation_steps 1 \
--val_check_interval 1.0 \
--max_iter 3