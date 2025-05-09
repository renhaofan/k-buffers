#!/bin/bash
#########################################################################
# File Name: run_rasterization.sh
# Author: steve
# E-mail: yqykrhf@163.com
# Created Time: Tue 07 Mar 2023 11:10:38 PM CST
# Brief: 
#########################################################################
echo "------Rasterization:, device $1---------"
export CUDA_VISIBLE_DEVICES=$1
#python main.py --config=configs/ablation/lego_pointnerf_1.txt
python main.py --config=configs/ablation/lego_pointnerf_2.txt
#python main.py --config=configs/ablation/lego_pointnerf_4.txt
#python main.py --config=configs/ablation/lego_pointnerf_8.txt
