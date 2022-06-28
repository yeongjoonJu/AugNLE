#!/bin/bash
DATE=$(date +"%Y-%m-%d")
CKPT_DIR="./ckpts"

CUDA_VISIBLE_DEVICES=0,1,2,3 python ../trainer.py \
--experiment_name $(date +%D-%T) \
--max_epochs 1 \
--ngpu 1 \
--warmup_steps 100 \
--checkpoints_dir ${CKPT_DIR} \
--img_size 224 \
--valid_anno_path "/media/storage/coco/VQA-X/annotated/vqaX_val.json" \
--train_anno_path "/media/storage/coco/VQA-X/annotated/vqaX_train.json" \
--pseudo_labels_pth "/media/storage/coco/VQA-X/annotated/vqaX_pseudo_labeling.json" \
--captioning_pth "/media/storage/coco/caption_data/nocaps/filtered_78.51_k-100_p-0.5_t-0.7_n_1.json" \
--image_dir "/media/storage/coco/image" \
--lm_backbone "distilgpt2" \
--train_batch_size 3 \
--learning_rate 2e-5 \
--dec_max_len 40 \
--gradient_accumulation_steps 1 \
--fewshot_ratio 0 \
--val_check_interval 1.0 \
--output_dir "outputs" \
--temperature 1 \
--iteration 10 \
