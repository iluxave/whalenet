#!/bin/bash
export CUDA_VISIBLE_DEVICES=''
eval python3 ~/ai/tf-models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=./mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.config \
    --checkpoint_dir=./google_object_detection/mask_rcnn_inception_resnet_v2_atrous_coco \
    --eval_dir=./google_object_detection/mask_rcnn_inception_resnet_v2_atrous_coco-eval

