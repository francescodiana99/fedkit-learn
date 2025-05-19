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

split_criterion="random"
n_tasks=2
n_local_steps=1
batch_size=32
optimizer="sgd"

results_dir="./results/seeds/$seed/linear/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer/reconstructed"
reconstructed_models_dir="./reconstructed_models/seeds/$seed/linear/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"
metadata_dir="./metadata/seeds/$seed/linear/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"

cmd="python run_linear_mb_aia.py   \
--metadata_dir $metadata_dir \
--seed $seed --device $device --sensitive_attribute smoker_yes \
--results_dir $results_dir \
--reconstructed_models_dir $reconstructed_models_dir \
--n_trials $n_trials \
--verbose "

echo $cmd
eval $cmd
