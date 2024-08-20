#!/bin/bash
cd ../../../scripts

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    mix_coefficient=10
else
    mix_coefficient=$2
fi

if [ -z "$3" ]; then
    n_rounds=100
else
    n_rounds=$3
fi

if [ -z "$4" ]; then
    n_trials=50
else
    n_trials=$4
fi

if [ -z "$5" ]; then
    seed="sgd"
else
    seed=$5
fi

device="cuda"
n_tasks=10
optimizer="adam"
state="louisiana"

cmd="python run_local_models_optimization.py --task_name income --optimizer $optimizer --batch_size $batch_size --device $device \
 --data_dir ./data/seeds/$seed/income/tasks/correlation/$state/$mix_coefficient/$n_tasks/all \
--logs_dir ./logs/seeds/$seed/income/$state/mixed/$mix_coefficient/$n_tasks/$batch_size/local/$optimizer \
--metadata_dir ./metadata/seeds/$seed/income/$state/mixed/$mix_coefficient/$n_tasks/$batch_size/1/sgd \
--log_freq 1 --save_freq 1 --num_rounds $n_rounds --seed $seed \
--model_config_path ../fedklearn/configs/income/$state/models/config_1.json --n_trials $n_trials \
--hparams_config_path ../fedklearn/configs/income/$state/hyperparameters/local/hp_space_local.json \
--local_models_dir ./local_models/seeds/$seed/income/$state/mixed/$mix_coefficient/10/$batch_size/optimized/$optimizer"

eval $cmd



