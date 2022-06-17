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
--load_ckpt_path ckpts/2022-06-17_02:23/epoch=03-val_loss=0.33.ckpt \
--top_p 0.7 \
--temperature 1.0