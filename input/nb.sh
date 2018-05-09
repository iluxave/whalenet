#!/bin/bash
export CUDA_VISIBLE_DEVICES=''

cd $(dirname $0)/..
jupyter notebook --no-browser --NotebookApp.iopub_data_rate_limit=0
