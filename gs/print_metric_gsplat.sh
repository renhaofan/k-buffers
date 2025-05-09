#!/bin/bash

# output_dir=results/benchmark/360_v2
# output_dir=results/benchmark/360_v2_absgrad_0.0008

# output_dir=results/bpcr_blender_20240808011842
# output_dir=results/bpcr_dtu_20240808024021
# output_dir=results/bpcr_scannet_20240808033435

# output_dir=results_neural/bpcr_blender_20240808012203

# output_dir=results_neural/bpcr_blender_20240808012203
# output_dir=results_neural/bpcr_dtu_20240808011943
# output_dir=results_neural/bpcr_scannet_20240808043845
# output_dir=results_neural/360_v2

############################### update object-centric scene with white_background for kfn ########################
# 3DGS update, 3DGS ours update
# output_dir=results/bpcr_dtu_20240811100855
# output_dir=results_neural/bpcr_dtu_opa_0.005_20240811171507

# output_dir=results_neural/bpcr_blender_opa_0.005_20240812012437

# 3dgs(ours)_black_bg_mask
# output_dir=results_neural/bpcr_blender_opa_0.005_20240811202458
# output_dir=results/bpcr_blender_20240811052927

# output_dir=results/bpcr_dtu_20240811100855
#output_dir=results_neural/bpcr_dtu_opa_0.005_20240812011058
output_dir=$2

#####  mask not correct ########
# output_dir=results/bpcr_dtu_20240811062558
# output_dir=results_neural/bpcr_dtu_opa_0.005_20240811065757
#####  mask not correct ########


# [psnr, ssim, lpips, ellipse_time, num_GS]
metric=$1
# step=$2
# step=60000
step=30000
# step=40000

sum=0
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

    ret=`cat $output_dir/$scene/stats/val_step$step.json | jq .$metric`
    echo $ret
    sum=$(echo "$sum + $ret" | bc)
    count=$((count + 1))
done
echo "--------------------------"

# 计算平均值
if [ $count -ne 0 ]; then
    average=$(echo "scale=8; $sum / $count" | bc)
    printf "Average: %.4f\n" $average
else
    echo "No scenes processed."
fi
