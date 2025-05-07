#!/bin/bash

# Save the current directory
original_dir=$(pwd)


cd ../../../scripts || exit

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
    device="cuda"
else
    device=$6
fi

if [ -z "$7" ]; then
    add_args=""
else
    add_args=$7
fi

state="full"
optimizer="sgd"
split_criterion="state"
n_task_samples=39133
n_tasks=51

# privacy parameters
delta=1e-5
clip=3e6
epsilon=1

model_config_path="../fedklearn/configs/income/full/models/net_config.json"
chkpts_dir="./chkpts/seeds/$seed/privacy/$epsilon/$delta/$clip/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/privacy/$epsilon/$delta/$clip/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"
metadata_dir="./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer"

cmd="python run_simulation.py \
--task_name income \
--test_frac 0.1 \
--scaler standard \
--optimizer $optimizer \
--learning_rate $lr \
--momentum 0.0  \
--weight_decay 0.0 \
--batch_size $batch_size \
--local_steps $n_local_steps \
--by_epoch \
--device $device \
--data_dir ./data/seeds/$seed/income/ \
--log_freq 5 \
--save_freq 1 \
--num_rounds $num_rounds \
--seed $seed \
--model_config_path $model_config_path \
--split_criterion $split_criterion \
--n_tasks $n_tasks \
--n_task_samples $n_task_samples \
--state $state \
--chkpts_dir $chkpts_dir \
--logs_dir $logs_dir \
--metadata_dir $metadata_dir \
--keep_proportions \
--use_dp \
--dp_epsilon $epsilon \
--dp_delta $delta \
--clip_norm $clip \
--num_active_rounds $num_active_rounds \
$add_args
"

echo $cmd
eval $cmd

cd $original_dir || exit