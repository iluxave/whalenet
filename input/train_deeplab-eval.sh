#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=''
. deeplab_settings
python3 ~/ai/tf-models/research/deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="${MODEL_VARIANT}" \
    ${XCEPTION_OPTIONS} \
    --eval_crop_size=1050 \
    --eval_crop_size=1600 \
    --save_summaries_images=True \
    --dataset="whalenet" \
    --checkpoint_dir=${TRAINDIR}/train \
    --eval_logdir=${TRAINDIR}/eval \
    --dataset_dir=${SCRIPT_DIR}/tfrecord
