MODEL=${MODEL:-facebook/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
EPS=${EPS:-5e-2}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-10000}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

LR=5e-3
TASK=RTE
SEED=0
RANK=4
P=4
# STEP_INTERVAL_v=1
STEP_INTERVAL_u=50
Tainer=GELOZO

case $TASK in
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

TAG=$Tainer-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL_u-$RANK-$P


echo $TAG
echo "Task: $TASK"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "RANK: $RANK"
echo "PKEEP: $P"
echo "STEP INTERVAL: $STEP_INTERVAL_u"
# echo "STEP INTERVAL: $STEP_INTERVAL_v"

export TRANSFORMERS_OFFLINE=1
gpu_no="${1:-7}"
de=${de:-""}
cmd="python"
if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 1999 --wait-for-client"
fi
CUDA_VISIBLE_DEVICES="$gpu_no" $cmd run_gelozo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/text/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $Tainer --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS \
    --train_as_classification \
    --step_u_interval $STEP_INTERVAL_u \
    --rank_r $RANK \
    --p_keep $P \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 





