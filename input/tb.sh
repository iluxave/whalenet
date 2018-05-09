#!/bin/bash
export CUDA_VISIBLE_DEVICES=''

exec tensorboard --logdir=./google_object_detection/

