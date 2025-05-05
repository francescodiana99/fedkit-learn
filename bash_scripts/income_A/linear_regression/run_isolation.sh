#!/bin/bash
cd ../../../scripts
# Check if batch_size and local_epochs are provided as command line arguments, otherwise use default values

original_dir=$(pwd)

if [ -z "$1" ]; then
    seed='42'
else
    seed=$1
fi

if [ -z "$2" ]; then
    device='cpu'
else
    device=$2
fi

# Hyperparameter options
attacked_round=99
num_rounds=50
n_tasks=51
local_epochs=1
batch_size=32
state="full"
n_task_samples="39133"
optimizer="sgd"
n_local_steps=1
split_criterion="state"
attacked_task=12 # this can be randomly selected from 0 to 50

metadata_dir="./metadata/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd"
logs_dir="./logs/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd/isolated"
iso_chkpts_dir="./chkpts/seeds/$seed/linear/income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/sgd/isolated"
# Define base command
base_cmd="python run_isolation.py \
--seed $seed \
--num_rounds $num_rounds \
--device $device \
--metadata_dir $metadata_dir \
--logs_dir $logs_dir \
--iso_chkpts_dir $iso_chkpts_dir \
--attacked_round $attacked_round \
--attacked_task $attacked_task"

# Run the command
echo "Running $base_cmd"
eval $base_cmd

cd $original_dir 
