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

batch_size=32
n_local_steps=1
optimizer="sgd"
n_tasks=51
n_task_samples=39133
#NOTE: if you change the number of local epochs, you need to adapt the keep_rounds_frac values
keep_rounds_frac="0. 0.05 0.10 0.20 0.5 1.0"

# privacy parameters (needed only for the paths)
delta=1e-5
clip=3e6
epsilon=1

attacked_task=5 # client id to attack, reduces the computation time. To optimize all tasks, remove this

metadata_dir="./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer/gb_aia"
learning_rates="100 1000 10000 100000 1000000"
results_path="./results/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer/gb_aia.json"

cmd="python run_gb_aia.py \
--num_rounds 100 \
--device $device \
--seed $seed \
--sensitive_attribute SEX \
--sensitive_attribute_type binary \
--keep_rounds_frac $keep_rounds_frac \
--metadata_dir $metadata_dir \
--logs_dir $logs_dir \
--learning_rate $learning_rates \
--results_path $results_path \
--keep_first_rounds \
--track_time \
--attacked_task $attacked_task "

echo "Running $cmd"
eval $cmd

cd $original_dir || exit
