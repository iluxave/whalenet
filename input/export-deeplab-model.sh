#!/bin/bash
function die() {
    echo $@
    exit 1
}
export CUDA_VISIBLE_DEVICES=''

. deeplab_settings
set -x
# get the latest checkpoint
checkpoint=$(ls google_object_detection/deeplab/${INIT_CHECKPOINT}/train/model.ckpt-*.index|sort -n|tail -n1)
[ -n "${checkpoint}" ] || die "Could not find a checkpoint"
checkpoint=${checkpoint/.index/}
step=$(basename ${checkpoint})
step=${step/model.ckpt-/}
echo step=$step
mkdir deeplab/export/$step/
python3 ~/ai/tf-models/research/deeplab/export_model.py \
  --logtostderr \
  --checkpoint_path="${checkpoint}" \
  --export_path="deeplab/export/$step/frozen_inference_graph.pb" \
  --model_variant="${MODEL_VARIANT}" \
  ${XCEPTION_OPTIONS} \
  --num_classes=2 \
  --crop_size=600 \
  --crop_size=1600 \
  --inference_scales=1.0
cp deeplab/export/$step/frozen_inference_graph.pb deeplab/export/
