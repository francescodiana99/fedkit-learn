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
    optimizer="sgd"
else
    optimizer=$3
fi


if [ -z "$4" ]; then
    seed=0
else
    seed=$4
fi

if [ -z "$5" ]; then
    device='cuda'
else
    device=$5
fi

n_tasks=51
n_task_samples=39133

# Define base command
base_cmd="python run_aia.py \
  --task_name income \
  --temperature 1.0 \
  --metadata_path ./metadata/seeds/$seed/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/$optimizer/federated.json \
  --data_dir ./data/seeds/$seed/income/tasks/state/full/$n_tasks/$n_task_samples/ \
  --split train \
  --optimizer sgd \
  --num_rounds 100 \
  --device cuda \
  --log_freq 1 \
  --seed $seed \
  --sensitive_attribute_type binary"

# Hyperparameter options
learning_rates=(100 1000 10000 100000 1000000)
sensitive_attributes=("SEX")
keep_rounds_frac=(0. 0.05 0.10 0.20 0.5 1.0)  # Add keep_rounds_frac values

# Nested loop for hyperparameter search
for lr in "${learning_rates[@]}"; do
  for attr in "${sensitive_attributes[@]}"; do
    for frac in "${keep_rounds_frac[@]}"; do
      full_cmd="$base_cmd --learning_rate $lr --sensitive_attribute $attr --keep_first_rounds --keep_rounds_frac $frac \
      --results_path ./results/seeds/$seed/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/$optimizer/aia_${lr}_${frac}.json \
      --logs_dir ./logs/seeds/$seed/income/full/$n_tasks/$n_task_samples/$batch_size/$local_epochs/$optimizer/aia_${lr}_${frac} "
      echo "Running with learning rate: $lr, sensitive attribute: $attr"
      eval "$full_cmd"
    done
  done
done
