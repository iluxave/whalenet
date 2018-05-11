#!/bin/bash
set -x
. deeplab_settings

exec python3 ~/ai/tf-models/research/deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=50000 \
    --train_split="train" \
    --model_variant="${MODEL_VARIANT}" \
    --train_crop_size=500 \
    --train_crop_size=600 \
    --dataset="whalenet" \
    --tf_initial_checkpoint=${CKPT_FILE} \
    --train_logdir=${TRAINDIR}/train \
    --dataset_dir=${SCRIPT_DIR}/tfrecord \
    --fine_tune_batch_norm=False \
    --train_batch_size=12 \
    --save_summaries_images=True \
    --last_layers_contain_logits_only=True \
    --initialize_last_layer=False \
    ${XCEPTION_OPTIONS} \
    --num_clones=${NUM_CLONES} \
    --base_learning_rate=.0001
