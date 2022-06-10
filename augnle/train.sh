#!/bin/bash
DATE=$(date +"%Y-%m-%d")
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0 python Trainer.py \
--experiment_name $(date +%D-%T) \
--max_epochs 30 \
--ngpu 1 \
--warmup_steps 100 \
--ckpt_dir ${CKPT_DIR} \
--img_size 224 \
--train_anno_path "../nle_anno/VQA-X/vqaX_train.json" \
--valid_anno_path "../nle_anno/VQA-X/vqaX_val.json" \
--image_dir "../images" \
--enc_max_len 512 \
--dec_max_len 64 \
--train_batch_size 8 \
--learning_rate 2e-5 \
--gradient_accumulation_steps 1 \
--fewshot_ratio 0.1 \
--prefix_len 50 \
--val_check_interval 1.0