
timestamp=$(date +"%Y%m%d%H%M%S")
RESULT_DIR=results_mcmc/bpcr_scannet_$timestamp

for SCENE in 0000 0043 0045;
do
    DATA_FACTOR=1

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=$1 python simple_trainer_mcmc_bpcr_scannet.py --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/scannet/$SCENE \
        --result_dir $RESULT_DIR/$SCENE/
    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=$1 python simple_trainer_mcmc_bpcr_scannet.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/scannet/$SCENE \
            --result_dir $RESULT_DIR/$SCENE \
            --ckpt $CKPT
    done
done


for SCENE in 0000 0043 0045;
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
