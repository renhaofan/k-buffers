#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
#scenes=$2

for scene in "${scenes[@]}"; do
    echo "------Fast Training: device $1, scene $scene-------"
    #python run_rasterize.py --config=configs/bpcr_mpn_tiny_fast_crop800_dim8/${scene}_pointnerf.txt
    python run_rasterize.py --config=configs/nerf_bpcr/${scene}_pointnerf.txt
    python main_fast_3.py --config=configs/nerf_bpcr/${scene}_pointnerf.txt
done
