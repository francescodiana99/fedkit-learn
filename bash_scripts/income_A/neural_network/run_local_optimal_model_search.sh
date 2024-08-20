#!/bin/bash

cd ../../../scripts

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    n_rounds=100
else
    n_rounds=$2
fi

if [ -z "$3" ]; then
    n_trials=50
else
    n_trials=$3
fi

if [ -z "$4" ]; then
    seed=0
else
    seed=$4
fi

if [ -z "$5" ]; then
    device="cuda"
else
    device=$5
fi


n_tasks=51
optimizer="adam"
n_task_samples=39133
state='full'
n_local_steps=1

cmd="python run_local_models_optimization.py --task_name income --optimizer $optimizer --batch_size $batch_size --device $device \
 --data_dir ./data/seeds/$seed/income/tasks/state/$state/$n_tasks/$n_task_samples \
--logs_dir ./logs/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/local/sgd \
--metadata_dir ./metadata/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd \
--log_freq 1 --save_freq 1 --num_rounds $n_rounds --seed $seed \
--model_config_path ../fedklearn/configs/income/$state/$n_tasks/$n_task_samples/models/config_1.json --n_trials $n_trials \
--hparams_config_path ../fedklearn/configs/income/$state/hyperparameters/local/hp_space_local.json \
--local_models_dir ./local_models/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/optimized/sgd "


echo $cmd
eval $cmd


