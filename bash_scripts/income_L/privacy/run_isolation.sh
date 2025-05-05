#!/bin/bash

cd ../../../scripts

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    local_epochs=1
else
    local_epochs=$2
fi

if [ -z "$3" ]; then
    mix_percentage=10
else
    mix_percentage=$3
fi

if [ -z "$4" ]; then
    learning_rate=0.01
else
    learning_rate=$4
fi

if [ -z "$5" ]; then
    seed="sgd"
else
    seed=$5
fi

if [ -z "$6" ]; then
    round=99
else
    round=$6
fi


if [ -z "$7" ]; then
    epsilon=0.1
else
    epsilon=$7
fi

if [ -z "$8" ]; then
    clip=1.0
else
    clip=$8
fi

if [ -z "$9" ]; then
    device='cuda'
else
    device=$9
fi
state='louisiana'
optimizer='sgd'

base_cmd="python run_isolation.py \
  --by_epoch \
  --data_dir ./data/seeds/$seed/income/tasks/correlation/$state/$mix_percentage/10/all \
  --task_name income \
  --split train \
  --metadata_dir ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/ \
  --optimizer sgd \
  --momentum 0. \
  --weight_decay 0.0 \
  --batch_size $batch_size \
  --num_epochs 50 \
  --device cuda \
  --log_freq 1 \
  --save_freq 1 \
  --seed $seed \
 --use_dp \
  --dp_epsilon $epsilon \
  --dp_delta 1e-5 \
  --clip_norm $clip"

cmd="$base_cmd --learning_rate $learning_rate --attacked_round $round --attacked_task 3 \
  --logs_dir ./logs/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/isolated/$round \
  --isolated_models_dir  ./isolated/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/$round"
echo "Running command: $cmd"
  # Execute the command
  $cmd

