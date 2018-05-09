#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=''
. deeplab_settings
python3 ~/ai/tf-models/research/deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=700 \
    --vis_crop_size=1600 \
    --dataset="whalenet" \
    --checkpoint_dir=${TRAINDIR}/train \
    --vis_logdir=${TRAINDIR}/vis \
    --dataset_dir=${SCRIPT_DIR}/tfrecord
