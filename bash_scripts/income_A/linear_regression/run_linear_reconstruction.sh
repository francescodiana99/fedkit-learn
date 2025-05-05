#!/bin/bash
cd ../../../scripts

if [ -z "$1" ]; then
    seed=42
else
    seed=$1
fi

if [ -z "$2" ]; then
    n_trials=10000
else
    n_trials=$2
fi

if [ -z "$3" ]; then
    device="cuda"
else
    device=$3
fi

split_criterion="state"
n_tasks=51
n_task_samples=39133
state="full"
n_local_steps=1
batch_size=32
optimizer="sgd"

results_dir="./results/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer/reconstructed"
reconstructed_models_dir="./reconstructed_models/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"
metadata_dir="./metadata/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd"

cmd="python run_linear_mb_aia.py   \
--metadata_dir $metadata_dir \
--seed $seed --device $device --sensitive_attribute SEX \
--results_dir $results_dir \
--reconstructed_models_dir $reconstructed_models_dir \
--n_trials $n_trials \
--verbose "

echo $cmd
eval $cmd
