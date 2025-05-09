GROW_GRAD2D=0.0008
RESULT_DIR=results/benchmark/360_v2_absgrad_$GROW_GRAD2D

for SCENE in bicycle bonsai counter flowers garden kitchen room stump treehill;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ] || [ "$SCENE" = "flowers" ] || [ "$SCENE" = "treehill" ]; then
        DATA_FACTOR=4
    else
        DATA_FACTOR=2
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=1 python simple_trainer.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --absgrad \
        --grow_grad2d $GROW_GRAD2D

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=1 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in bicycle bonsai counter flowers garden kitchen room stump treehill;
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
