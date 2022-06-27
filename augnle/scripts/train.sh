#!/bin/bash
DATE=$(date +"%Y-%m-%d")
CKPT_DIR="./ckpts"
# --weight_decay 0.04 \

CUDA_VISIBLE_DEVICES=0,1,2 python tune.py \
--experiment_name xl_adaf_b8x3x2_lr0.3_ci_ep30_t2 \
--max_epochs 30 \
--ngpu 3 \
--warmup_steps 0 \
--ckpt_dir ${CKPT_DIR} \
--train_anno_path "../nle_anno/VQA-X/vqaX_train.json" \
--valid_anno_path "../nle_anno/VQA-X/vqaX_val.json" \
--train_batch_size 8 \
--learning_rate 0.3 \
--gradient_accumulation_steps 2 \
--fewshot_ratio 1.0 \
--prefix_len 100 \
--val_check_interval 1.0 \
--optimizer adafactor