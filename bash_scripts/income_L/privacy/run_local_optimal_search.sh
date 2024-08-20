#!/bin/bash

cd ../../../scripts

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    mix_coefficient=10
else
    mix_coefficient=$2
fi

if [ -z "$3" ]; then
    n_rounds=100
else
    n_rounds=$3
fi

if [ -z "$4" ]; then
    n_trials=50
else
    n_trials=$4
fi

if [ -z "$5" ]; then
    seed="sgd"
else
    seed=$5
fi

if [ -z "$6" ]; then
    epsilon=0.1
else
    epsilon=$6
fi

if [ -z "$7" ]; then
    clip=1.0
else
    clip=$7
fi

if [ -z "$8" ]; then
    device="cuda"
else
    device=$8
fi


device=$device
n_tasks=10
optimizer="adam"
state="louisiana"
local_epochs=1

cmd="python run_local_models_optimization.py --task_name income --optimizer $optimizer --batch_size $batch_size --device $device \
 --data_dir ./data/seeds/$seed/income/tasks/correlation/$state/$mix_coefficient/10/all \
--logs_dir ./logs/seeds/$seed/income/$state/mixed/$mix_coefficient/$n_tasks/$batch_size/local/sgd \
--metadata_dir  ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_coefficient/10/$batch_size/$local_epochs/sgd/ \
--log_freq 1 --save_freq 1 --num_rounds $n_rounds --seed $seed --use_dp --dp_epsilon $epsilon  --dp_delta 1e-5 --clip_norm $clip --optimized_task 3 \
--model_config_path ../fedklearn/configs/income/$state/models/net_config.json --n_trials $n_trials \
--hparams_config_path ../fedklearn/configs/income/$state/hyperparameters/local/hp_space_local.json \
--local_models_dir ./local_models/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_coefficient/10/$batch_size/optimized/$optimizer"

eval $cmd


