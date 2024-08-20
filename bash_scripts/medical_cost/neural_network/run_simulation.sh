#!/bin/bash
cd ../../../scripts

force_flag=false

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
    lr=0.5
else
    lr=$3
fi

if [ -z "$4" ]; then
    num_rounds=100
else
    num_rounds=$4
fi

if [ -z "$5" ]; then
    n_tasks=2
else
    n_tasks=$5
fi

if [ -z "$6" ]; then
    optimizer="sgd"
else
    optimizer=$6
fi

if [ -z "$7" ]; then
    split_criterion='random'
else
    split_criterion=$7
fi

if [ -z "$7" ]; then
    seed=0
else
    seed=$8
fi

if [ -z "$9" ]; then
    device="cuda"
else
    device=$9
fi

if [ "${10}" == "force_generation" ]; then
    force_flag=true
fi

if [ "${11}" == "download" ]; then
    down_flag=true
fi
cmd="python run_simulation.py --task_name medical_cost --test_frac none --scaler standard --optimizer $optimizer --learning_rate $lr \
--momentum 0.0 --weight_decay 0.0 --batch_size $batch_size --device $device  --log_freq 10 \
  --save_freq 1 --num_rounds $num_rounds --seed $seed --model_config_path ../fedklearn/configs/medical_cost/models/net_config.json \
  --split_criterion $split_criterion --device $device --n_tasks $n_tasks --by_epoch\
   --data_dir ./data/seeds/$seed/medical_cost --test_frac 0.1"

if $force_flag; then
    cmd="$cmd --force_generation"
fi
  echo $cmd

if $down_flag; then
    cmd="$cmd --download"

  full_cmd=" $cmd
  --chkpts_dir ./chkpts/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/$optimizer  \
  --logs_dir ./logs/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/$optimizer  \
  --metadata_dir ./metadata/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/$optimizer  \
  "
  eval $full_cmd





