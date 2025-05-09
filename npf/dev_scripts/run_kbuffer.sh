#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

version=$2
scenes=("dtu_110" "lego_pointnerf" "scan_00")

for scene in "${scenes[@]}"; do
    echo "------Fast Training K-BUFFERS: device $1, scene $scene, version $version-------"
    python run_rasterize.py --config=configs/debug/v$version/${scene}.txt
    python main_fast_k.py --config=configs/debug/v$version/${scene}.txt
done