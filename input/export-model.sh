#!/bin/bash
function die() {
    echo $@
    exit 1
}
export CUDA_VISIBLE_DEVICES=''

. ./mask_model

# get the latest checkpoint
checkpoint=$(ls google_object_detection/mask/${MODEL}/model.ckpt-*.index|sort -n|tail -n1)
[ -n "${checkpoint}" ] || die "Could not find a checkpoint"
checkpoint=${checkpoint/.index/}
step=$(basename ${checkpoint})
step=${step/model.ckpt-/}
echo step=$step
python3 ~/ai/tf-models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path google_object_detection/mask/${MODEL}/pipeline.config \
    --trained_checkpoint_prefix $checkpoint \
    --output_directory mask_inference_graph/$step

cp mask_inference_graph/$step/frozen_inference_graph.pb mask_inference_graph/
