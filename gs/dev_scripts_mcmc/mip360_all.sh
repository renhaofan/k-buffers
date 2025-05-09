
timestamp=$(date +"%Y%m%d%H%M%S")
RESULT_DIR=results_neural_mcmc/360_v2_$timestamp

for SCENE in bicycle bonsai counter flowers garden kitchen room stump treehill;
# for SCENE in flowers treehill;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ] || [ "$SCENE" = "flowers" ] || [ "$SCENE" = "treehill" ]; then
        DATA_FACTOR=4
    else
        DATA_FACTOR=2
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=$1 python simple_neural_trainer_mcmc.py --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --app_opt \
        --nn_comp \
        --nn_comp_type kfn

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=$1 python simple_neural_trainer_mcmc.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in bicycle bonsai counter flowers garden kitchen room stump treehill;
# for SCENE in flowers treehill;
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
