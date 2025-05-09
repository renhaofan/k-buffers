#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

#scenes=("scan_00" "scan_43" "scan_45")
scenes=$2

for scene in "${scenes[@]}"; do
    echo "------Fast Training: device $1, scene $scene-------"
    python run_rasterize.py --config=configs/scannet_fast_2_frepcr/${scene}.txt
    python main_fast_3.py --config=configs/scannet_fast_2_frepcr/${scene}.txt
done
