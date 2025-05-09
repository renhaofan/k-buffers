
timestamp=$(date +"%Y%m%d%H%M%S")

RESULT_DIR=results_neural_mcmc/bpcr_dtu_opa_$timestamp

for SCENE in dtu_110 dtu_114 dtu_118;
do
    DATA_FACTOR=1

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=$1 python simple_neural_trainer_mcmc_bpcr_dtu.py --disable_viewer --data_factor $DATA_FACTOR \
        --data_dir data/dtu/$SCENE \
        --result_dir $RESULT_DIR/$SCENE/ \
        --app_opt \
        --nn_comp \
        --nn_comp_type kfn

    # run eval and render
    # for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    # do
    #     CUDA_VISIBLE_DEVICES=$1 python simple_neural_trainer_bpcr_dtu.py --disable_viewer --data_factor $DATA_FACTOR \
    #         --data_dir data/dtu/$SCENE \
    #         --result_dir $RESULT_DIR/$SCENE \
    #         --ckpt $CKPT
    # done
done


for SCENE in dtu_110 dtu_114 dtu_118;
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
