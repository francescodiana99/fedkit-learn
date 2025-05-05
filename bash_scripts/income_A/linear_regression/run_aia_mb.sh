#!/bin/bash

cd ../../../scripts

original_dir=$(pwd)

if [ -z $1 ]; then
    device='cuda'
else
    device=$1
fi

if [ -z $2 ]; then
    seed=42
else
    seed=$2
fi

if [ -z $3 ]; then
    attacked_round=99
else
    attacked_round=$3
fi

attacked_task=12
batch_size=32
n_local_steps=1
optimizer="sgd"
n_tasks=51
n_task_samples=39133
state="full"
split_criterion="state"
active_rounds="0 9 49"
metadata_dir="./metadata/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd"
results_dir="./results/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"

cmd="python run_mb_aia.py \
--device $device \
--seed $seed \
--sensitive_attribute SEX \
--sensitive_attribute_type binary \
--metadata_dir $metadata_dir \
--results_dir $results_dir \
--active_rounds $active_rounds \
--attacked_round $attacked_round \
--attacked_task $attacked_task \
--use_oracle"

echo $cmd
eval $cmd

cd $original_dir 
