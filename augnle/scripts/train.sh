#!/bin/bash
DATE=$(date +"%Y-%m-%d")
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0,1,2 python Trainer.py \
--experiment_name $(date +%D-%T) \
--max_epochs 10 \
--ngpu 3 \
--warmup_steps 1000 \
--ckpt_dir ${CKPT_DIR} \
--img_size 224 \
--train_anno_path "../nle_anno/VQA-X/vqaX_train.json" \
--valid_anno_path "../nle_anno/VQA-X/vqaX_val.json" \
--image_dir "../images" \
--enc_max_len 512 \
--dec_max_len 64 \
--train_batch_size 4 \
--learning_rate 0.2 \
--gradient_accumulation_steps 1 \
--fewshot_ratio 1.0 \
--prefix_len 100 \
--val_check_interval 1.0 \
--weight_decay 0.04 \
--optimizer adamw \
--load_ckpt_path ckpts/2022-06-17_02:23/epoch=03-val_loss=0.33.ckpt