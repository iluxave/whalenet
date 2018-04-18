#!/bin/bash
python3 ~/ai/tf-models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path google_object_detection/mask_rcnn_inception_resnet_v2_atrous_coco/pipeline.config \
    --trained_checkpoint_prefix google_object_detection/mask_rcnn_inception_resnet_v2_atrous_coco/model.ckpt-10000 \
    --output_directory mask_inference_graph.pb
