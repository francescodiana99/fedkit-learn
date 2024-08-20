#!/bin/bash
cd ../../../scripts

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    seed=32
else
    seed=$2
fi

if [ -z "$3" ]; then
    n_trials=32
else
    n_trials=$3
fi

split_criterion="random"
n_tasks=10
state="louisiana"
n_local_steps=1
device='cuda'

cmd="python evaluate_linear_reconstruction.py  --data_dir data/seeds/42/linear_income/tasks/$split_criterion/$state/$n_tasks/all/ \
--metadata_dir metadata/seeds/$seed/linear_income/$state/$split_criterion/$n_tasks/$batch_size/$n_local_steps/sgd/  \
 --seed $seed --device $device --n_trials $n_trials --sensitive_attribute SEX \
 --results_dir results/seeds/$seed/linear_income_louisiana/$state/$split_criterion/$n_tasks/$batch_size/$n_local_steps/sgd/reconstructed \
 --reconstructed_models_dir ./reconstructed_models/seeds/$seed/linear_income/$state/$split_criterion/$n_tasks/$batch_size/$n_local_steps/sgd/
 --track_time --verbose "

echo $cmd
eval $cmd
