
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

batch_size=32
n_local_steps=1
optimizer="sgd"
n_tasks=51
n_task_samples=39133
active_rounds="0 9 49"

# privacy parameters (needed only for the paths)
delta=1e-5
clip=3e6
epsilon=1

attacked_task=5 # task to attack, remove if you want to attack all tasks

metadata_dir="./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"
results_dir="./results/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"

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
--use_oracle "

echo $cmd
eval $cmd

cd $original_dir 
