
timestamp=$(date +"%Y%m%d%H%M%S")
RESULT_DIR=results_mcmc/bpcr_blender_$timestamp

for SCENE in chair drums ficus hotdog lego materials mic ship;
do
    DATA_FACTOR=1

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=$1 python simple_trainer_mcmc_bpcr_blender.py --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/nerf_synthetic/$SCENE \
        --result_dir $RESULT_DIR/$SCENE/
    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=$1 python simple_trainer_mcmc_bpcr_blender.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/nerf_synthetic/$SCENE \
            --result_dir $RESULT_DIR/$SCENE \
            --ckpt $CKPT
    done
done


for SCENE in chair drums ficus hotdog lego materials mic ship;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done
