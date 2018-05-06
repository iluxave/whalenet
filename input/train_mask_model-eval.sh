#!/bin/bash
export CUDA_VISIBLE_DEVICES=''
. ./mask_model
eval python3 ~/ai/tf-models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=./${MODEL}.config \
    --checkpoint_dir=./google_object_detection/mask/${MODEL} \
    --eval_dir=./google_object_detection/mask/${MODEL}-eval

