
#!/bin/bash
cd ../../../scripts
# Check if batch_size and local_epochs are provided as command line arguments, otherwise use default values

original_dir=$(pwd)

if [ -z "$1" ]; then
    seed=42
else
    seed=$1
fi

if [ -z "$2" ]; then
    device='cuda'
else
    device=$2
fi

# Hyperparameter options
attacked_round=99
num_rounds=50
n_tasks=51
n_task_samples=39133
local_epochs=1
batch_size=32

# privacy parameters (needed only for the paths)
delta=1e-5
clip=3e6
epsilon=1

attacked_task=5 # client id to optimize, reduces the computation time. To optimize all clients, remove this

# Define base command
base_cmd="python run_isolation.py \
--seed $seed \
--num_rounds $num_rounds \
--device $device \
--metadata_dir ./metadata/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/sgd \
--logs_dir ./logs/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/sgd/isolated \
--iso_chkpts_dir ./chkpts/seeds/$seed/privacy/$epsilon/$delta/$clip/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/sgd/isolated \
--attacked_round $attacked_round \
--attacked_task $attacked_task"

# Run the command
echo "Running $base_cmd"
eval $base_cmd

cd $original_dir 
