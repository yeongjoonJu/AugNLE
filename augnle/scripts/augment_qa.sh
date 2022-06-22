#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python generate_qa_pair.py \
--anno_path ../nle_anno/VQA-X/vqaX_val.json \
--object_label_path ../nle_anno/VQA-X/obj_val_labels.json \
--batch_size 32 \
--prefix_len 100 \
--load_ckpt_path ckpts/adaf_b18_lr0.3_ci_ep20_obj4/epoch=19-val_loss=0.262.ckpt \
--top_k 100 \
--top_p 0.5 \
--num_return_sequences 3 \
--temperature 0.7