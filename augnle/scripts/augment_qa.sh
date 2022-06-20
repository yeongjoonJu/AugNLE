#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python Trainer.py \
--ngpu 1 \
--img_size 224 \
--train_anno_path "../nle_anno/VQA-X/vqaX_train.json" \
--valid_anno_path "../nle_anno/VQA-X/vqaX_val.json" \
--image_dir "../images" \
--enc_max_len 512 \
--dec_max_len 64 \
--eval_batch_size 32 \
--prefix_len 100 \
--load_ckpt_path ckpts/sep_prompt_adamw_b12_lr0.2/epoch=02-val_loss=0.29.ckpt \
--top_p 0.7 \
--temperature 1.0