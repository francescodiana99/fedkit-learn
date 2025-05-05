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
    mix=0.1
else
    mix=$6
fi

if [ -z "$7" ]; then
    device='cuda'
else
    device=$7
fi

if [ -z "$8" ]; then
    add_args=""
else
    add_args=$8
fi

state="louisiana"
optimizer="sgd"
n_tasks=10
model_config_path="../fedklearn/configs/income/$state/models/net_config.json"
split_criterion="correlation"
momentum=0.0
weight_decay=0.0
data_dir="./data/seeds/$seed/income"

mix_scaled=$(echo "$mix * 100" | bc -l | awk '{print int($1)}')
chkpts_dir="./chkpts/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/$optimizer"
metadata_dir="./metadata/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/$optimizer"


cmd="python run_simulation.py \
--task_name income \
--test_frac 0.1 \
--scaler standard \
--optimizer $optimizer \
--learning_rate $lr \
--momentum $momentum  \
--weight_decay $weight_decay \
--batch_size $batch_size 
--local_steps $n_local_steps \
--by_epoch \
--device $device \
--data_dir $data_dir  \
--log_freq 5 \
--save_freq 1 \
--num_rounds $num_rounds \
--seed $seed \
--model_config_path $model_config_path \
--split_criterion $split_criterion \
--n_tasks $n_tasks \
--state $state \
--mixing_coefficient $mix \
--chkpts_dir $chkpts_dir \
--logs_dir $logs_dir \
--metadata_dir $metadata_dir \
$add_args
"

echo $cmd
eval $cmd

cd $original_dir || exit
