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
    attacked_round="0 9 49 99"
else
    attacked_round=$3
fi

if [ -z $4 ];then
    seed=0
else
    seed=$4
fi

if [ -z $5 ];then
    attacked_client=0
else
    attacked_client=$5
fi

if [ -z $6 ];then
    device=0
else
    device=$6
fi

n_tasks=10
local_epochs=1
state="louisiana"
optimizer='sgd'
active_rounds=(9 49)

for active_round in "${active_rounds[@]}"; do
    script="python evaluate_aia.py --task_name linear_income \
    --models_metadata_path  ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/federated.json --device $device --seed $seed \
    --results_dir ./results/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/active --batch_size $batch_size \
    --sensitive_attribute SEX --sensitive_attribute_type binary --split train \
    --reference_models_metadata_path ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/local_trajectories.json \
    --active_models_metadata_path ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/active_$attacked_round.json \
    --data_dir ./data/seeds/42/linear_income/tasks/random/$state/$n_tasks/all  \
    --models_config_metadata_path ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$local_epochs/$optimizer/model_config.json \
    --verbose --attacked_rounds $attacked_round --active_round $active_round --attacked_client $attacked_client"
    echo $script
echo "Active Round: $active_round | Attacked: round: $attacked_round"
    eval $script
done
