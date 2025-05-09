#!/bin/bash
echo "------device $1-------"
export CUDA_VISIBLE_DEVICES=$1

python profile_bpcr.py --config=configs/ablation/lego_pointnerf_1.txt