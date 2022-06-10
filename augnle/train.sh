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
--train_anno_dir "/media/storage/coco/VQA-X/annotated/" \
--image_dir "/media/storage/coco/" \
--train_batch_size 3 \
--learning_rate 2e-5 \
--gradient_accumulation_steps 1 \
--fewshot_ratio 0.1
