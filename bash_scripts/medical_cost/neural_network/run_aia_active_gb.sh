#!/bin/bash
# Check if batch_size and local_epochs are provided as command line arguments, otherwise use default values

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
    n_tasks=2
else
    n_tasks=$3
fi

if [ -z "$4" ]; then
    optimizer="sgd"
else
    optimizer=$4
fi

if [ -z "$5" ]; then
    split_criterion="random"
else
    split_criterion=$5
fi

if [ -z "$6" ]; then
    seed=0
else
    seed=$6
fi

if [ -z "$7" ]; then
    active_round="sgd"
else
    active_round=$7
fi

if [ -z "$8" ]; then
    device='cuda'
else
    device=$8
fi

# Define base command
base_cmd="python run_aia.py \
  --task_name medical_cost \
  --temperature 1.0 \
  --metadata_path ./metadata/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/$optimizer/federated.json \
  --data_dir ./data/seeds/$seed/medical_cost/tasks/$split_criterion/$n_tasks \
  --split train \
  --optimizer sgd \
  --num_rounds 100 \
  --device $device \
  --logs_dir ./logs/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/$optimizer/isolated/${active_round}/aia_${lr}_${frac} \
  --log_freq 1 \
  --seed $seed \
  --sensitive_attribute_type binary\
  --active_server \
  --local_models_metadata_path ./metadata/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/$optimizer/isolated_trajectories_${active_round}.json \
  --isolated"

# Hyperparameter options
learning_rates=(100 1000 10000 100000  1000000)
sensitive_attributes=("smoker_yes")
keep_rounds_frac=(0. 0.05 0.10 0.20 0.5 1.0)  # Add keep_rounds_frac values

# Nested loop for hyperparameter search
for lr in "${learning_rates[@]}"; do
  for attr in "${sensitive_attributes[@]}"; do
    for frac in "${keep_rounds_frac[@]}"; do
      full_cmd="$base_cmd --learning_rate $lr --sensitive_attribute $attr --keep_rounds_frac $frac --keep_first_rounds\
      --results_path ./results/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$local_epochs/$optimizer/isolated/${active_round}/aia_${lr}_${frac}.json"
      echo "Running with learning rate: $lr, sensitive attribute: $attr, keep_rounds_frac: $frac, keep_first_rounds: True"
      eval "$full_cmd"
    done
  done
done


