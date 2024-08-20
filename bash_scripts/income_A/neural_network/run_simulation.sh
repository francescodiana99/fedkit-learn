#!/bin/bash

cd ../../scripts

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

if [ "${6}" == "force_generation" ]; then
    force_flag=true
fi

if [ "${7}" == "download" ]; then
    download=true
fi

device="cuda"
state="full"
optimizer="sgd"
split_criterion="state"
n_task_samples=39133
n_tasks=51


cmd="python run_simulation.py --task_name income --test_frac 0.1 --scaler standard --optimizer $optimizer --learning_rate $lr \
--momentum 0.0  --weight_decay 0.0 --batch_size $batch_size --local_steps $n_local_steps --by_epoch --device $device \
--data_dir ./data/seeds/$seed/income/  --log_freq 5 --save_freq 1 --num_rounds $num_rounds --seed $seed \
--model_config_path ../fedklearn/configs/income/$state/$n_tasks/$n_task_samples/models/net_config.json --split_criterion $split_criterion \
--n_tasks $n_tasks --n_task_samples $n_task_samples --state $state\
  --chkpts_dir ./chkpts/fedkit-learn/chkpts/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer \
  --logs_dir ./logs/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer \
  --metadata_dir ./metadata/seeds/$seed/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer \
  --keep_proportions \
  "
if force_flag; then
    cmd="$cmd --force_generation"
fi
if download; then
    cmd="$cmd --download"

  echo $cmd
  eval $cmd

