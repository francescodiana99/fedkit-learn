#!/bin/bash
cd ../../../scripts
force_flag=false
download=false

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
    download=true
fi

device="cuda"
state="full"
n_local_steps=1
optimizer="sgd"
split_criterion="state"
n_task_samples=39133
n_tasks=51

cmd="python run_simulation.py --task_name linear_income --test_frac 0.1 --scaler standard --optimizer $optimizer --learning_rate $lr \
--momentum 0.0  --weight_decay 0.0 --batch_size $batch_size --local_steps $n_local_steps --by_epoch --device $device \
--data_dir ./data/linear_income/seeds/$seed  --log_freq 10 --save_freq 1 --num_rounds $num_rounds --seed $seed \
--model_config_path ../fedklearn/configs/income/$state/$n_tasks/$n_task_samples/models/linear_config.json \
 --split_criterion $split_criterion \
--n_tasks $n_tasks --n_task_samples $n_task_samples --state $state\
  --chkpts_dir ./chkpts/seeds/$seed/linear_income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer \
  --logs_dir ./logs/seeds/$seed/linear_income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer \
  --metadata_dir ./metadata/seeds/$seed/linear_income/$state/$n_tasks/$n_task_samples/$batch_size/$n_local_steps/$optimizer \
  --keep_proportions --compute_local_models "
echo $force_flag
echo $download

if [ "$force_flag" == true ]; then
    echo $force_flag
    cmd="$cmd --force_generation "
fi

if [ "$download" == true ]; then
    cmd="$cmd --download "
fi
  echo $cmd
  eval $cmd
