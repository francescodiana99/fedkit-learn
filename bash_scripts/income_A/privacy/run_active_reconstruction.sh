#!/bin/bash

original_dir=$(pwd)

cd ../../../scripts

if [ -z "${1}" ]; then
    n_trials=10
else
    n_trials=$2
fi

if [ -z "$2" ]; then
    seed="0"
else
    seed=$2
fi

if [ -z "$3" ]; then
    device="cuda"
else
    device=$3
fi

attacked_round=99

batch_size=256
n_local_steps=1
optimizer="sgd"
n_tasks=51
n_task_samples=39133

# privacy parameters (needed only for the paths)
delta=1e-5
clip=3e6
epsilon=1

attacked_task=5 # task to attack, remove if you want to attack all tasks

metadata_dir="./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd"

logs_dir="./logs/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd/active/$attacked_round"
hparams_config_path="../fedklearn/configs/income/full/hyperparameters/hp_space_attack.json"

cmd="python run_active_simulation.py \
--device $device \
--log_freq 5 \
--save_freq 1 \
--num_rounds 10 \
--seed $seed \
--attacked_round $attacked_round \
--optimize_hyperparams \
--n_trials $n_trials \
--metadata_dir $metadata_dir \
--logs_dir  $logs_dir \
--hparams_config_path $hparams_config_path \
--attacked_task $attacked_task"

echo  "Running $cmd"
eval $cmd

cd $original_dir || exit
