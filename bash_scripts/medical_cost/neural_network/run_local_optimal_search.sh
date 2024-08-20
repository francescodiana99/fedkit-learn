#!/bin/bash

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    n_rounds=100
else
    n_rounds=$2
fi

if [ -z "$3" ]; then
    n_trials=50
else
    n_trials=$3
fi

if [ -z "$4" ]; then
    seed=0
else
    seed=$4
fi

if [ -z "$5" ]; then
    device='cuda'
else
    device=$5
fi

n_tasks=2
optimizer="adam"
split_criterion='random'
n_local_steps=1

cmd="python run_local_models_optimization.py --task_name medical_cost --optimizer $optimizer --batch_size $batch_size --device $device \
 --data_dir ./data/seeds/$seed/medical_cost/tasks/$split_criterion/$n_tasks \
--logs_dir ./logs/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/local/$optimizer \
--metadata_dir ./metadata/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/sgd \
--log_freq 1 --save_freq 1 --num_rounds $n_rounds --seed $seed \
--model_config_path ../fedklearn/configs/medical_cost/models/config_1.json --n_trials $n_trials \
--hparams_config_path ../fedklearn/configs/medical_cost/hyperparams/local/hp_space_local.json \
--local_models_dir ./local_models/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/optimized/$optimizer"

echo $cmd
eval $cmd