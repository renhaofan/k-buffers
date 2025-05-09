#!/bin/bash
#########################################################################
echo "------ device $1---------"
echo "------Rasterization: $2---------"

export CUDA_VISIBLE_DEVICES=$1
python run_rasterize.py --config=$2
