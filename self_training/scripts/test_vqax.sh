#!/bin/bash
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0 python trainer.py \
--mode test \
--output_dir outputs/base_b32 \
--load_ckpt_path ckpts/nlx_gpt_base_b32/epoch=29-val_loss=0.695.ckpt \
--eval_batch_size 1 \
--ngpu 1 \
--img_size 224 \
--nle_test_anno_path "../nle_anno/VQA-X/vqaX_test.json" \
--nle_test_image_dir "../images/val2014" \
--top_k 0 \
--top_p 0.9 \
--temperature 1.0