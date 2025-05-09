#!/bin/bash
#########################################################################
# File Name: run_rasterization.sh
# Author: steve
# E-mail: yqykrhf@163.com
# Created Time: Tue 07 Mar 2023 11:10:38 PM CST
# Brief: 
#########################################################################
echo "-----device $1---------"
export CUDA_VISIBLE_DEVICES=$1
#python run_rasterize.py --config=configs/bpcr_raw/chair_pointnerf.txt
#python run_rasterize.py --config=configs/bpcr_raw/materials_pointnerf.txt
#python run_rasterize.py --config=configs/bpcr_raw/mic_pointnerf.txt
# python main.py --config=configs/bpcr_raw/materials_pointnerf.txt

python run_rasterize.py --config=configs/nerf_bpcr/lego_pointnerf.txt
python main_fast_3.py --config=configs/nerf_bpcr/lego_pointnerf.txt

#python run_rasterize.py --config=configs/nerf_frepcr/materials_pointnerf.txt
#python main_fast_3.py --config=configs/nerf_frepcr/materials_pointnerf.txt
