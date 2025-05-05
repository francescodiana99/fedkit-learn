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
    active_round=0
else
    active_round=$5
fi

if [ -z $6 ];then
    mix_percentage=10
else
    mix_percentage=$6
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
    device="cuda"
else
    device=$9
fi

state="louisiana"
optimizer="sgd"

    script="python evaluate_aia.py --task_name income \
    --models_metadata_path ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/federated.json --device $device --seed $seed \
    --results_dir ./results/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/active --batch_size $batch_size \
    --sensitive_attribute SEX --sensitive_attribute_type binary --split train \
    --active_models_metadata_path ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/active_$attacked_round.json \
    --data_dir ./data/seeds/$seed/income/tasks/correlation/louisiana/$mix_percentage/10/all/ \
    --models_config_metadata_path ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_percentage/10/$batch_size/$local_epochs/$optimizer/model_config.json \
    --verbose --attacked_rounds $attacked_round --active_round $active_round --attacked_client 3"
    echo $script
echo "Active Round: $active_round | Attacked: round: $attacked_round"
eval $script

