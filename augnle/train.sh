#!/bin/bash
DATE=$(date +"%Y-%m-%d")
CKPT_DIR="/media/storage/checkpoints/AUG_NLX/${DATE}/"
mkdir ${CKPT_DIR}

python Trainer.py \
--output_dir outputs/ \
--experiment_name $(date +%D-%T) \
--max_epochs 5 \
--ngpu 1 \
--warmup_steps 100 \
--ckpt_path "/media/storage/checkpoints/AUG_NLX/data/" \
--checkpoints_dir ${CKPT_DIR} \
--img_size 224 \
--annotation_data_dir "/media/storage/coco/VQA-X/annotated/" \
--coco_data_dir "/media/storage/coco/" \
--input_max_seq_length 500 \
--output_max_seq_length 60 \
--train_batch_size 3 \
--learning_rate 2e-5 \
--gradient_accumulation_steps 1 \
--prompting \
--task_A \
--num_workers 3 \
--fewshot 0.1 \