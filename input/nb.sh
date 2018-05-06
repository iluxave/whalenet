#!/bin/bash
export CUDA_VISIBLE_DEVICES=''

jupyter notebook --no-browser --NotebookApp.iopub_data_rate_limit=0
