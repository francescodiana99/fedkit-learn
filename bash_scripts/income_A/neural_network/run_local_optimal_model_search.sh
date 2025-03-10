#!/bin/bash

cd ../../../scripts

original_dir=$(pwd)

if [ -z "$1" ]; then
    n_rounds=100
else
    n_rounds=$1
fi

if [ -z "$2" ]; then
    n_trials=50
else
    n_trials=$2
fi

if [ -z "$3" ]; then
    seed=0
else
    seed=$3
fi

if [ -z "$4" ]; then
    device="cuda"
else
    device=$4
fi

n_tasks=51
optimizer="adam"
n_task_samples=39133
state='full'
n_local_steps=1
metadata_dir="./metadata/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd"
logs_dir="./logs/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer/local"
local_chkpts_dir="./chkpts/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer/local"
hparams_config_path=".../fedklearn/configs/income/full/hyperparameters/hp_space_local.json"

cmd="python run_local_models_optimization.py \
--optimizer $optimizer \
--num_rounds $n_rounds \
--local_chkpts_dir $local_chkpts_dir \
--device $device \
--seed $seed \
--metadata_dir $metadata_dir \
--logs_dir $logs_dir \
--n_trials $n_trials \
--hparams_config_path $hparams_config_path"

echo $cmd
eval $cmd

cd $original_dir || exit
