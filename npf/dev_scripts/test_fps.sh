#!/bin/bash
echo "------device $1-------"
export CUDA_VISIBLE_DEVICES=$1

# python inference_bpcr.py --config=configs/ablation/lego_pointnerf_1.txt
# python inference_mpn_rdmp.py --config=configs/ablation/lego_pointnerf_8.txt
python inference_kfn_fast.py --config=configs/ablation/lego_pointnerf_8.txt
# python inference_mpn_rdmp.py --config=configs/lego_pp.txt
