#!/bin/bash
set -x
. deeplab_settings
exec python3 ~/ai/tf-models/research/deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=50000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --dataset="whalenet" \
    --tf_initial_checkpoint=${DEEPDIR}/${INIT_CHECKPOINT}/model.ckpt \
    --train_logdir=${TRAINDIR}/train \
    --dataset_dir=${SCRIPT_DIR}/tfrecord \
    --fine_tune_batch_norm=False \
    --train_batch_size=8 \
    --save_summaries_images=True \
    --initialize_last_layer=True
