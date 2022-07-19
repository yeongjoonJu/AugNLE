#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python generate_qa_pair.py \
# --object_label_paths ../captioning_data/coco/obj_labels/coco_train_2014.json \
# --qa_save_dir ../captioning_data/coco/annotations/train2014 \
# --batch_size 64 \
# --prefix_len 100 \
# --load_ckpt_path ckpts/xl_adaf_b8x3_lr0.3_ci_ep30_t2/epoch=28-val_loss=0.2580.ckpt \
# --top_k 100 \
# --top_p 0.7 \
# --num_return_sequences 1 \
# --temperature 1.0

# CUDA_VISIBLE_DEVICES=1 python generate_qa_pair.py \
# --object_label_paths ../captioning_data/coco/obj_labels/coco_val_2014.json \
# --qa_save_dir ../captioning_data/coco/annotations/val2014 \
# --batch_size 64 \
# --prefix_len 100 \
# --load_ckpt_path ckpts/xl_adaf_b8x3_lr0.3_ci_ep30_t2/epoch=28-val_loss=0.2580.ckpt \
# --top_k 100 \
# --top_p 0.7 \
# --num_return_sequences 1 \
# --temperature 1.0

CUDA_VISIBLE_DEVICES=2 python generate_qa_pair.py \
--object_label_paths ../captioning_data/nocaps/obj_labels/nocaps.json \
--qa_save_dir ../captioning_data/nocaps/annotations \
--batch_size 64 \
--prefix_len 100 \
--load_ckpt_path ckpts/xl_adaf_b8x3_lr0.3_ci_ep30_t2/epoch=28-val_loss=0.2580.ckpt \
--top_k 100 \
--top_p 0.7 \
--num_return_sequences 1 \
--temperature 1.0

CUDA_VISIBLE_DEVICES=2 python generate_qa_pair.py \
--object_label_paths ../captioning_data/flickr30k/obj_labels/flickr.json \
--qa_save_dir ../captioning_data/flickr30k/annotations \
--batch_size 64 \
--prefix_len 100 \
--load_ckpt_path ckpts/xl_adaf_b8x3_lr0.3_ci_ep30_t2/epoch=28-val_loss=0.2580.ckpt \
--top_k 100 \
--top_p 0.7 \
--num_return_sequences 1 \
--temperature 1.0