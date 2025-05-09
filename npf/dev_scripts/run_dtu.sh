#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

#scenes=("dtu_110" "dtu_114" "dtu_118")
scenes=$2

for scene in "${scenes[@]}"; do
    echo "------Fast Training: device $1, scene $scene-------"
    #python run_rasterize.py --config=configs/dtu_bpcr/${scene}.txt
    #python main_fast_3.py --config=configs/dtu_bpcr/${scene}.txt
    python main_fast_sh.py --config=configs/dtu_bpcr/${scene}.txt
done
