#!/bin/bash
cd ../../scripts
SCRIPT_DIR='../../scripts'

force_flag=false
download=false

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    n_local_steps=1
else
    n_local_steps=$2
fi

if [ -z "$3" ]; then
    lr=0.5
else
    lr=$3
fi

if [ -z "$4" ]; then
    seed=42
else
    seed=$4
fi

if [ "$5" == "force_generation" ]; then
    force_flag=true
fi

if [ "$6" == "download" ]; then
    download=true
fi



device="cpu"
split_criterion="random"
optimizer="sgd"
n_tasks=2
num_rounds=300

cmd="python run_simulation.py --task_name linear_medical_cost --test_frac 0.1 --scaler standard --optimizer $optimizer --learning_rate $lr \
--momentum 0.0 --weight_decay 0.0 --batch_size $batch_size --device $device  --log_freq 10 \
  --save_freq 1 --num_rounds $num_rounds --seed $seed --model_config_path ../fedklearn/configs/medical_cost/models/linear_config.json \
  --split_criterion $split_criterion --device $device --n_tasks $n_tasks --by_epoch\
   --data_dir ./data/seeds/$seed/linear_medical_cost"

  full_cmd=" $cmd
  --chkpts_dir ./chkpts/seeds/$seed/linear_medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/$optimizer  \
  --logs_dir ./logs/seeds/$seed/linear_medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/$optimizer  \
  --metadata_dir ./metadata/seeds/$seed/linear_medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/$optimizer  \
  "

echo "$force_flag"
echo "$download"
if [ "$force_flag" == true ]; then
    full_cmd="$full_cmd --force_generation"
fi

if [ "$download" == true ]; then
    full_cmd="$full_cmd --download"
fi
echo $full_cmd
eval $full_cmd
