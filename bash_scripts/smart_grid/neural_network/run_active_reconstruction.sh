#!/bin/bash

original_dir=$(pwd)

cd ../../../scripts


if [ -z "$1" ]; then
    attacked_round=99
else
    attacked_round=$1
fi

if [ -z "${2}" ]; then
    n_trials=50
else
    n_trials=$2
fi

if [ -z "$3" ]; then
    seed="42"
else
    seed=$3
fi

if [ -z "$4" ]; then
    device="cuda"
else
    device=$4
fi

optimizer="sgd"
n_tasks=2
n_local_steps=1
batch_size=32
logs_dir="./logs/seeds/$seed/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer/active/$attacked_round"
metadata_dir="./metadata/seeds/$seed/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"
hparams_config_path="../fedklearn/configs/smart_grid/hyperparameters/hp_space_attack.json"

cmd="python run_active_simulation.py \
--device $device \
--log_freq 5 \
--save_freq 1 \
--num_rounds 50 \
--seed $seed \
--attacked_round $attacked_round \
--optimize_hyperparams \
--n_trials $n_trials \
--metadata_dir $metadata_dir \
--logs_dir  $logs_dir \
--hparams_config_path $hparams_config_path"

if [ -n "$attacked_task" ]; then
    cmd="$cmd --attacked_task $attacked_task"
fi

echo  "Running $cmd"
eval $cmd

cd $original_dir 
