#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=''
. deeplab_settings
python3 ~/ai/tf-models/research/deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --save_summaries_images=True \
    --dataset="whalenet" \
    --checkpoint_dir=${TRAINDIR}/train \
    --eval_logdir=${TRAINDIR}/eval \
    --dataset_dir=${SCRIPT_DIR}/tfrecord
