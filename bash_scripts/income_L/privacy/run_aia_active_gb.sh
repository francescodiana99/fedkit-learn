
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
    mix_percentage=10
else
    mix_percentage=$3
fi

if [ -z "$4" ]; then
    state="louisiana"
else
    state=$4
fi

if [ -z "$5" ]; then
    optimizer="sgd"
else
    optimizer=$5
fi

if [ -z "$6" ]; then
    seed="sgd"
else
    seed=$6
fi

if [ -z "$7" ]; then
    epsilon=5
else
    epsilon=$7
fi

if [ -z "$8" ]; then
    clip=10
else
    clip=$8
fi

if [ -z "$9" ]; then
    active_round=99
else
    active_round=$9
fi

if [ -z "${10}" ]; then
    device=cpu
else
    device=${10}
fi

n_tasks=10

# Define base command
base_cmd="python run_aia.py \
  --task_name income \
  --temperature 1.0 \
  --metadata_path ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/federated.json \
  --data_dir ./data/seeds/$seed/income/tasks/correlation/$state/$mix_percentage/10/all \
  --split train \
  --optimizer sgd \
  --num_rounds 100 \
  --device $device \
  --log_freq 1 \
  --seed $seed \
  --sensitive_attribute_type binary \
  --track_time \
  --isolated \
  --active_server \
  --local_models_metadata_path ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/isolated_trajectories_${active_round}.json \
  --attacked_task 3"

# Hyperparameter options
learning_rates=(100 1000 10000 100000 1000000)
sensitive_attributes=("SEX")
keep_rounds_frac=(0. 0.05 0.10 0.20 0.5 1.0)  # Add keep_rounds_frac values

# Nested loop for hyperparameter search
for lr in "${learning_rates[@]}"; do
  for attr in "${sensitive_attributes[@]}"; do
    for frac in "${keep_rounds_frac[@]}"; do
      full_cmd="$base_cmd --learning_rate $lr --sensitive_attribute $attr --keep_first_rounds --keep_rounds_frac $frac \
      --results_path ./results/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/isolated/${active_round}/aia_${lr}_${frac}.json \
      --logs_dir ./logs/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/isolated/${active_round}/aia_${lr}_${frac} "
      echo "Running with learning rate: $lr, sensitive attribute: $attr"
      eval "$full_cmd"
    done
  done
done
