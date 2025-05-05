#!/bin/bash
cd ../../../scripts

original_dir=$(pwd)

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    n_local_steps=1
else
    n_local_steps=$2
fi

if [ -z "$3" ]; then
    lr=0.05
else
    lr=$3
fi

if [ -z "$4" ]; then
    num_rounds=100
else
    num_rounds=$4
fi

if [ -z "$5" ]; then
    seed=0
else
    seed=$5
fi

if [ -z "$6" ]; then
    device='cuda'
else
    device=$6
fi

if [ -z "$7" ]; then
    add_args=""
else
    add_args=$7
fi

optimizer="sgd"
n_local_steps=1
state="louisiana"
n_tasks=10
data_dir="./data/seeds/$seed/income"
split_criterion="random"
state="louisiana"

chkpts_dir="./chkpts/seeds/$seed/linear/income/$state/$split_criterion/$n_tasks/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/linear/income/$state/$split_criterion/$n_tasks/$batch_size/$n_local_steps/$optimizer"
metadata_dir="./metadata/seeds/$seed/linear/income/$state/$split_criterion/$n_tasks/$batch_size/$n_local_steps/$optimizer"
model_config_path="../fedklearn/configs/income/$state/models/linear_config.json"

cmd="python run_simulation.py \
--task_name income \
--scaler standard \
--optimizer $optimizer \
--momentum 0.0 \
--weight_decay 0.0 \
--batch_size $batch_size \
--local_steps $n_local_steps \
--by_epoch \
--device $device \
--data_dir $data_dir \
--log_freq 10 \
--save_freq 1 \
--num_rounds $num_rounds \
--seed $seed \
--model_config_path $model_config_path \
--split_criterion $split_criterion \
--n_tasks $n_tasks \
--state $state \
--chkpts_dir $chkpts_dir \
--logs_dir $logs_dir \
--metadata_dir $metadata_dir \
--learning_rate $lr \
--scale_target \
$add_args"

echo $cmd
eval $cmd

cd $original_dir
