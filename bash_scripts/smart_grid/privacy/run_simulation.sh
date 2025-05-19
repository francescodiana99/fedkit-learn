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
n_tasks=2
model_config_path="../fedklearn/configs/smart_grid/models/net_config.json"
split_criterion="random"
momentum=0.0
weight_decay=0.0
data_dir="./data/seeds/$seed/smart_grid"


# privacy parameters
delta=1e-6
clip=5e5
epsilon=1
num_active_rounds=50

chkpts_dir="./chkpts/seeds/$seed/privacy/$epsilon/$delta/$clip/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/privacy/$epsilon/$delta/$clip/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"
metadata_dir="./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"


cmd="python run_simulation.py \
--task_name smart_grid \
--test_frac 0.1 \
--scaler standard \
--optimizer $optimizer \
--learning_rate $lr \
--momentum $momentum \
--weight_decay $weight_decay \
--batch_size $batch_size \
--device $device  \
--log_freq 5 \
--save_freq 1 --num_rounds $num_rounds \
--seed $seed \
--model_config_path $model_config_path \
--split_criterion $split_criterion \
--n_tasks $n_tasks \
--by_epoch \
--data_dir $data_dir \
--chkpts_dir $chkpts_dir \
--logs_dir $logs_dir \
--metadata_dir $metadata_dir \
--use_dp \
--dp_epsilon $epsilon \
--dp_delta $delta \
--clip_norm $clip \
--num_active_rounds $num_active_rounds \
$add_args "


echo "Running command: $cmd"
eval $cmd

cd $original_dir || exit
