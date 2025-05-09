#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

version=$2
baseline=frepcr

#scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
#for scene in "${scenes[@]}"; do
#    echo "------Fast Training K-BUFFERS: device $1, scene $scene-------"
#    python run_rasterize.py --config=configs/nerf_$baseline/${scene}_pointnerf.txt
#    python main_fast_k.py --config=configs/nerf_$baseline/${scene}_pointnerf.txt
#done

#scenes=("dtu_110" "dtu_114" "dtu_118")
#for scene in "${scenes[@]}"; do
#    echo "------Fast Training: device $1, scene $scene-------"
#    python run_rasterize.py --config=configs/dtu_$baseline/${scene}.txt
#    python main_fast_k.py --config=configs/dtu_$baseline/${scene}.txt
#done

scenes=("scan_00" "scan_43" "scan_45")
for scene in "${scenes[@]}"; do
    echo "------Fast Training: device $1, scene $scene-------"
    python run_rasterize.py --config=configs/scannet_$baseline/${scene}.txt
    python main_fast_k.py --config=configs/scannet_$baseline/${scene}.txt
done
