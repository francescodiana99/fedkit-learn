#!/bin/bash

cd ../../../scripts

original_dir=$(pwd)

if [ -z "$1" ]; then
    seed=0
else
    seed=$1
fi

if [ -z "$2" ]; then
    device='cuda'
else
    device=$2
fi


n_tasks=2
n_local_steps=1
batch_size=32
learning_rates="100 1000 10000 100000 1000000"
sensitive_attribute="smoker_yes"
optimizer="sgd"
metadata_dir="./metadata/seeds/$seed/medical_cost/$n_tasks/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/medical_cost/$n_tasks/$batch_size/$n_local_steps/$optimizer/gb_aia"
results_path="./results/seeds/$seed/medical_cost/$n_tasks/$batch_size/$n_local_steps/$optimizer/gb_aia.json"

#NOTE: if you change the number of local epochs, you need to adapt the keep_rounds_frac values
keep_rounds_frac="0. 0.05 0.10 0.20 0.5 1.0"  # Add keep_rounds_frac values

cmd="python run_gb_aia.py \
--num_rounds 100 \
--device $device \
--seed $seed \
--sensitive_attribute $sensitive_attribute \
--sensitive_attribute_type binary \
--keep_rounds_frac $keep_rounds_frac \
--metadata_dir $metadata_dir \
--logs_dir $logs_dir \
--learning_rate $learning_rates \
--results_path $results_path \
--keep_first_rounds \
--track_time"

echo "Running $cmd"
eval $cmd

cd $original_dir 
