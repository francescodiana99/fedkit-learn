#!/bin/bash
cd ../../../scripts


#!/bin/bash
force_flag=false

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    lr=0.05
else
    lr=$2
fi

if [ -z "$3" ]; then
    num_rounds=100
else
    num_rounds=$3
fi

if [ -z "$4" ]; then
    seed=0
else
    seed=$4
fi

if [ "$5" == "force_generation" ]; then
    force_flag=true
fi

if [ "$6" == "download" ]; then
    down_flag=true
fi

echo $force_flag

device="cuda"
optimizer="sgd"
n_local_steps=1
state="louisiana"
n_tasks=10

cmd="python run_simulation.py --task_name linear_income --test_frac 0.1 --scaler standard --optimizer $optimizer --learning_rate $lr \
--momentum 0.0  --weight_decay 0.0 --batch_size $batch_size --local_steps $n_local_steps --by_epoch --device $device \
--data_dir ./data/seeds/42/linear_income  --log_freq 10 --save_freq 1 --num_rounds $num_rounds --seed $seed \
--model_config_path ../fedklearn/configs/income/$state/models/linear_config.json --split_criterion random \
--n_tasks $n_tasks --state $state --compute_local_models \
--chkpts_dir ./chkpts/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer \
--logs_dir ./logs/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer \
--metadata_dir ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer \
--local_models_dir ./fedkit-learn/local_models/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer  "


if [ $force_flag == true ]; then
    echo $force_flag
    cmd="$cmd --force_generation"
fi

if [ $down_flag == true ]; then
    echo $down_flag
    cmd="$cmd --download"
fi

echo $cmd
eval $cmd

