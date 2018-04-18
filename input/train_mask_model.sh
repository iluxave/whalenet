#!/bin/bash
exec python3 ~/ai/tf-models/research/object_detection/train.py --logtostderr --pipeline_config_path=./mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.config --train_dir=./google_object_detection/mask/mask_rcnn_inception_resnet_v2_atrous_coco 
