#!/bin/bash

. ./mask_model

exec python3 ~/ai/tf-models/research/object_detection/train.py --logtostderr --pipeline_config_path=./${MODEL}.config --train_dir=./google_object_detection/mask/${MODEL} $@
