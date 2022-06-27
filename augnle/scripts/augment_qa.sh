#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python generate_qa_pair.py \
# --object_label_paths ../captioning_data/coco/obj_labels/coco_train_2014.json ../captioning_data/coco/obj_labels/coco_val_2014.json \
# --qa_save_dir ../captioning_data/coco/annotations \
# --batch_size 64 \
# --prefix_len 100 \
# --load_ckpt_path ckpts/adaf_b12x3_lr0.3_ci_ep20_t3/epoch=18-val_loss=0.2551.ckpt \
# --top_k 100 \
# --top_p 0.5 \
# --num_return_sequences 1 \
# --temperature 0.7

CUDA_VISIBLE_DEVICES=1 python generate_qa_pair.py \
--object_label_paths ../captioning_data/nocaps/obj_labels/nocaps.json \
--qa_save_dir ../captioning_data/nocaps/annotations \
--batch_size 64 \
--prefix_len 100 \
--load_ckpt_path ckpts/adaf_b18_lr0.3_ci_ep20_obj4/epoch=18-val_loss=0.2551.ckpt \
--top_k 100 \
--top_p 0.5 \
--num_return_sequences 1 \
--temperature 0.7

# CUDA_VISIBLE_DEVICES=2 python generate_qa_pair.py \
# --object_label_paths ../captioning_data/flickr30k/obj_labels/flickr.json \
# --qa_save_dir ../captioning_data/flickr30k/annotations \
# --batch_size 64 \
# --prefix_len 100 \
# --load_ckpt_path ckpts/adaf_b12x3_lr0.3_ci_ep20_t3/epoch=18-val_loss=0.2551.ckpt \
# --top_k 100 \
# --top_p 0.5 \
# --num_return_sequences 1 \
# --temperature 0.7