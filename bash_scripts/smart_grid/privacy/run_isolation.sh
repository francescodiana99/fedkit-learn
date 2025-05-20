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
n_tasks=2
local_epochs=1
batch_size=32
optimizer="sgd"
n_local_steps=1

# Privacy parameters (needed only for the paths)
delta=1e-6
clip=5e5
epsilon=1

metadata_dir="./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer"
logs_dir="./logs/seeds/$seed/privacy/$epsilon/$delta/$clip/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer/isolated"
iso_chkpts_dir="./chkpts/seeds/$seed/privacy/$epsilon/$delta/$clip/smart_grid/$n_tasks/$batch_size/$n_local_steps/$optimizer/isolated"

# Define base command
base_cmd="python run_isolation.py \
--seed $seed \
--num_rounds $num_rounds \
--device $device \
--metadata_dir $metadata_dir \
--logs_dir $logs_dir \
--iso_chkpts_dir $iso_chkpts_dir \
--attacked_round $attacked_round"

# Run the command
echo "Running $base_cmd"
eval $base_cmd

cd $original_dir 
