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

if [ -z $4 ]; then
    mix_scaled=10
else
    mix_scaled=$4
fi

batch_size=32
n_local_steps=1
optimizer="sgd"
n_tasks=10
state="louisiana"
active_rounds="0 9 49"
metadata_dir="./metadata/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/$optimizer"
results_dir="./results/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/$optimizer"

cmd="python run_mb_aia.py \
--device $device \
--seed $seed \
--sensitive_attribute SEX \
--sensitive_attribute_type binary \
--metadata_dir $metadata_dir \
--results_dir $results_dir \
--active_rounds $active_rounds \
--attacked_round $attacked_round \
--use_oracle "

echo $cmd
eval $cmd

cd $original_dir 
