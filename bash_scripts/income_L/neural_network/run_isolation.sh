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
    device=(99)
else
    device=$6
fi

attacked_rounds=(99)
state='louisiana'

base_cmd="python run_isolation.py \
  --by_epoch \
  --data_dir ./data/seeds/$seed/income/tasks/correlation/$state/$mix_percentage/10/all \
  --task_name income \
  --split train \
  --metadata_dir ./metadata/seeds/$seed/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/sgd \
  --optimizer sgd \
  --momentum 0. \
  --weight_decay 0.0 \
  --batch_size $batch_size \
  --num_epochs 50 \
  --device $device \
  --logs_dir ./logs/seeds/$seed/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/sgd/isolated \
  --log_freq 1 \
  --save_freq 1 \
  --seed $seed "

for round in "${attacked_rounds[@]}"; do
  # Construct command with current hyperparameters
  cmd="$base_cmd --learning_rate $learning_rate --attacked_round $round \
  --isolated_models_dir ./isolated/seeds/$seed/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$round/sgd/ "
  echo "Running command: $cmd"
  # Execute the command
  $cmd
done
