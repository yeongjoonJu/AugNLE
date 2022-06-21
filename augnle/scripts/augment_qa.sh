#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python generate_qa_pair.py \
--anno_path ../nle_anno/VQA-X/vqaX_val.json \
--object_label_path ../nle_anno/VQA-X/obj_val_labels.json \
--batch_size 32 \
--prefix_len 100 \
--load_ckpt_path ckpts/sep_prompt_adaf_b18_lr0.3_obj2_ci_no_proj_ep20/epoch=18-val_loss=0.215.ckpt \
--top_k 1000 \
--top_p 0.9 \
--temperature 1.0