cd ../../../scripts

#!/bin/bash
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
    device="cuda"
else
    device=$7
fi


    script="python evaluate_aia.py --task_name income \
    --models_metadata_path ./metadata/seeds/$seed/income/louisiana/mixed/$mix_percentage/10/$batch_size/$local_epochs/sgd/federated.json --device $devi$    --results_dir ./results/seeds/$seed/income/louisiana/mixed/$mix_percentage/10/$batch_size/$local_epochs/sgd/active --batch_size $batch_size \
    --sensitive_attribute SEX --sensitive_attribute_type binary --split train \
    --reference_models_metadata_path ./metadata/seeds/$seed/income/louisiana/mixed/$mix_percentage/10/32/1/sgd/local_trajectories.json \
    --active_models_metadata_path ./metadata/seeds/$seed/income/louisiana/mixed/$mix_percentage/10/$batch_size/$local_epochs/sgd/active_$attacked_round$    --data_dir ./data/seeds/$seed/income/tasks/correlation/louisiana/$mix_percentage/10/all/ \
    --models_config_metadata_path ./metadata/seeds/$seed/income/louisiana/mixed/$mix_percentage/10/$batch_size/$local_epochs/sgd/model_config.json \
    --verbose --attacked_rounds $attacked_round --active_round $active_round"
    echo $script
echo "Active Round: $active_round | Attacked: round: $attacked_round"
    eval $script


