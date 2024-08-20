
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
    learning_rate=0.01
else
    learning_rate=$3
fi

if [ -z "$4" ]; then
    device='cuda'
else
    device=$4
fi

# Hyperparameter options
attacked_rounds=(99)
n_tasks=51
n_task_samples=39133

# Define base command
base_cmd="python run_isolation.py \
  --by_epoch \
  --data_dir ./data/income/tasks/state/full/$n_tasks/$n_task_samples/ \
  --task_name income \
  --split train \
  --metadata_dir ./metadata/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/sgd \
  --optimizer sgd \
  --momentum 0. \
  --weight_decay 0.0 \
  --batch_size $batch_size \
  --num_epochs 50 \
  --device cuda \
  --logs_dir ./logs/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/sgd/isolated \
  --log_freq 1 \
  --save_freq 1 \
  --seed 42 "

# Iterate over finetune rounds
for round in "${attacked_rounds[@]}"; do
  # Construct command with current hyperparameters
  cmd="$base_cmd --learning_rate $learning_rate --attacked_round $round \
  --isolated_models_dir ./isolated/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/sgd/$round "
  echo "Running command: $cmd"
  # Execute the command
  $cmd
done