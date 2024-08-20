#!/bin/bash

cd ../../../scripts
# Check if batch_size and local_epochs are provided as command line arguments, otherwise use default values
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
    learning_rate=1
else
    learning_rate=$3
fi

if [ -z "$4" ]; then
    optimizer='sgd'
else
    optimizer=$4
fi

if [ -z "$5" ]; then
    split_criterion='random'
else
    split_criterion=$5
fi

if [ -z "$6" ]; then
    seed=0
else
    seed=$6
fi

if [ -z "$7" ]; then
    device='cuda'
else
    device=$7
fi

# Hyperparameter options
attacked_rounds=(99)
n_tasks=2


# Define base command
base_cmd="python run_isolation.py \
  --by_epoch
  --data_dir ./data/seeds/$seed/medical_cost/tasks/$split_criterion/$n_tasks \
  --task_name medical_cost \
  --split train \
  --metadata_dir ./metadata/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/sgd \
  --optimizer $optimizer \
  --momentum 0. \
  --weight_decay 0.0 \
  --batch_size $batch_size \
  --num_epochs 50 \
  --learning_rate $learning_rate \
  --device $device \
  --log_freq 10 \
  --save_freq 1 \
  --seed $seed"

# Iterate over finetune rounds
for round in "${attacked_rounds[@]}"; do
  # Construct command with current hyperparameters
  cmd="$base_cmd  --attacked_round $round \
    --isolated_models_dir  ./isolated/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/$round/$optimizer/ \
    --logs_dir ./logs/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/sgd/isolated/$round"
  echo "Running command: $cmd"
  $cmd
  # Execute the command
done
