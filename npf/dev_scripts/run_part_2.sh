#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

# scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
scenes=("lego" "materials" "mic" "ship")

for scene in "${scenes[@]}"; do
    echo "------Fast Training: device $1, scene $scene-------"
    # python run_rasterize.py --config=configs/nerf_bpcr/${scene}_pointnerf.txt
    python main_fast_3.py --config=configs/nerf_bpcr/${scene}_pointnerf.txt
done