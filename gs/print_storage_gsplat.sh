#!/bin/bash

# output_dir=results/benchmark/360_v2
# output_dir=results/benchmark/360_v2_absgrad_0.0008

# output_dir=results/bpcr_blender_20240808011842
# output_dir=results/bpcr_dtu_20240808024021
# output_dir=results/bpcr_scannet_20240808033435


# output_dir=results_neural/bpcr_blender_20240808012203
# output_dir=results_neural/bpcr_dtu_20240808011943
# output_dir=results_neural/bpcr_scannet_20240808043845

# output_dir=results_neural/bpcr_blender_opa_0.002_20240809014726
# output_dir=results_mcmc/bpcr_blender_20240809073839



############################### update object-centric scene with white_background for kfn ########################
# output_dir=results/bpcr_blender_20240811052927
# output_dir=results_neural/bpcr_blender_opa_0.005_20240811052819

output_dir=$2

# output_dir=results/bpcr_dtu_20240811062558


# [psnr, ssim, lpips, ellipse_time, num_GS]
metric=$1
# step=60000
step=30000

total_size=0
count=0

step=$((step - 1))
echo "=========================="
echo "$output_dir $metric $step"
echo "=========================="

# scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
# for scene in "${scenes[@]}"; do
for scene in $(find "$output_dir" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort); do
# for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
    echo -n "$scene "

    ret=$(du -sh "$output_dir/$scene/ckpts/ckpt_${step}_rank0.pt" | cut -f1)
    echo $ret

     # 转换单位（假设 du 输出是 K,M,G 等单位）
    size=$(echo $ret | awk '
        /K/ { print $1 * 1024; next }
        /M/ { print $1 * 1024 * 1024; next }
        /G/ { print $1 * 1024 * 1024 * 1024; next }
        { print $1 }
    ')
    # 累加总大小
    total_size=$(($total_size + $size))
    # 计数器加1
    count=$((count + 1))
done
echo "--------------------------"

# 计算平均值（MB）
if [ $count -ne 0 ]; then
    avg_size_bytes=$(($total_size / $count))
    avg_size_mb=$(echo "scale=2; $avg_size_bytes / (1024 * 1024)" | bc)
    echo "Average size: $avg_size_mb MB"
else
    echo "No files found."
fi
