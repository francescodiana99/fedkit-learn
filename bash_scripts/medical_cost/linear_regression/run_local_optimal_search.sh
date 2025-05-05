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

n_tasks=2
optimizer="adam"
n_local_steps=1

metadata_dir="./metadata/seeds/$seed/linear/medical_cost/$n_tasks/$batch_size/$n_local_steps/sgd"
logs_dir="./logs/seeds/seeds/$seed/linear/medical_cost/$n_tasks/$batch_size/$n_local_steps/sgd/local"
local_chkpts_dir="./chkpts/seeds/$seed/linear/medical_cost/$n_tasks/$batch_size/$n_local_steps/sgd/local"
hparams_config_path="../fedklearn/configs/medical_cost/hyperparameters/hp_space_local_linear.json"

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

cd $original_dir 
