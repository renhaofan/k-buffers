RESULT_DIR=results/benchmark/tandt

for SCENE in drjohnson playroom train truck;
do
    DATA_FACTOR=1

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=2 python simple_trainer.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/tandt/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=2 python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/tandt/$SCENE/ \
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
