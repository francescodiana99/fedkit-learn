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

if [ -z $3 ]; then
    attacked_round=99
else
    attacked_round=$3
fi

if [ -z $4 ]; then
    seed=99
else
    seed=$4
fi

if [ -z $5 ]; then
    device='cuda'
else
    device=$5
fi

active_rounds=(9 49)
local_epochs=1

for active_round in "${active_rounds[@]}"; do
  script="python evaluate_aia.py --task_name medical_cost \
  --models_metadata_path ./metadata/seeds/$seed/medical_cost/random/2/$batch_size/local_epochs/$local_epochs/sgd/federated.json --device $device --seed $seed \
  --results_dir ./results/seeds/$seed/medical_cost/random/2/$batch_size/local_epochs/$local_epochs/sgd/active --batch_size $batch_size \
  --sensitive_attribute smoker_yes --sensitive_attribute_type binary --split train \
  --reference_models_metadata_path ./metadata/seeds/42/medical_cost/random/2/$batch_size/local_epochs/$local_epochs/sgd/local_trajectories.json \
  --active_models_metadata_path ./metadata/seeds/$seed/medical_cost/random/2/$batch_size/local_epochs/$local_epochs/sgd/active_$attacked_round.json \
  --data_dir ./data/seeds/$seed/medical_cost/tasks/random/2/ \
  --models_config_metadata_path ./metadata/seeds/$seed/medical_cost/random/2/$batch_size/local_epochs/$local_epochs/sgd/model_config.json \
  --verbose --attacked_rounds $attacked_round --active_round $active_round"
  echo $script
  eval $script

done