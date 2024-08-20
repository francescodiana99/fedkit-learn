#!/bin/bash
# Check if batch_size and local_epochs are provided as command line arguments, otherwise use default values
if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    seed="42"
else
    seed=$2
fi

if [ -z "$3" ]; then
    device="cuda"
else
    device=$3
fi

n_tasks=10
local_epochs=1
state="louisiana"
optimizer='sgd'


# Define base command
base_cmd="python run_aia.py \
  --task_name linear_income \
  --temperature 1.0 \
  --metadata_path ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/federated.json \
  --data_dir ./data/seeds/42/linear_income/tasks/random/$state/$n_tasks/all \
  --split train \
  --optimizer sgd \
  --num_rounds 100 \
  --device $device \
  --log_freq 1 \
  --seed $seed \
  --sensitive_attribute_type binary \
  --track_time"

# Hyperparameter options
learning_rates=(100 1000 10000 100000 1000000)
sensitive_attributes=("SEX")
keep_rounds_frac=(0. 0.017 0.033 0.067 0.167 0.33 0.5 1)  # Add keep_rounds_frac values

# Nested loop for hyperparameter search
for lr in "${learning_rates[@]}"; do
  for attr in "${sensitive_attributes[@]}"; do
    for frac in "${keep_rounds_frac[@]}"; do
      full_cmd="$base_cmd --learning_rate $lr --sensitive_attribute $attr --keep_first_rounds --keep_rounds_frac $frac \
      --results_path ./results/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/aia_${lr}_${frac}.json \
      --logs_dir ./logs/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/aia_${lr}_${frac} "
      echo "Running with learning rate: $lr, sensitive attribute: $attr"
      eval "$full_cmd"
    done
  done
done

