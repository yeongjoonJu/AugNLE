#!/bin/bash
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0 python trainer.py \
--mode test \
--output_dir outputs/fn_base_e30 \
--load_ckpt_path ckpts/base_nocap_flickr_b64_e30_lr2e-5/epoch=21-student_val_loss=1.093.ckpt \
--eval_batch_size 1 \
--ngpu 1 \
--img_size 224 \
--nle_test_anno_path "../nle_anno/VQA-X/vqaX_test.json" \
--nle_test_image_dir "../images/val2014" \
--top_k 0 \
--top_p 0.9 \
--temperature 1.0