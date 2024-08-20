#!/bin/bash


cd ../../../scripts


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
    num_rounds=100
else
    num_rounds=$4
fi

if [ -z "$5" ]; then
    n_tasks=2
else
    n_tasks=$5
fi

if [ -z "$6" ]; then
    alpha=0.05
else
    alpha=$6
fi

if [ -z "$7" ]; then
    beta1=0.05
else
    beta1=$7
fi

if [ -z "$8" ]; then
    beta2=0.05
else
    beta2=$8
fi

if [ -z "$9" ]; then
    attacked_round=99
else
    attacked_round=$9
fi

if [ -z "${10}" ]; then
    split_criterion=10
else
    split_criterion=${10}
fi

if [ -z "${11}" ]; then
    n_trials=10
else
    n_trials=${11}
fi

if [ -z "${12}" ]; then
    seed=0
else
    seed=${12}
fi

if [ -z "${13}" ]; then
    device="cuda"
else
    device=${13}
fi

cmd="python run_active_simulation.py --task_name medical_cost --test_frac none --scaler standard --optimizer sgd --learning_rate $lr \
--momentum 0.0 --weight_decay 0.0 --batch_size $batch_size --device $device  --log_freq 10 \
  --save_freq 1 --num_rounds $num_rounds --seed $seed --model_config_path ../fedklearn/configs/medical_cost/models/config_1.json \
 --device $device --n_tasks $n_tasks --by_epoch\
   --data_dir ./data/seeds/$seed/medical_cost/tasks/$split_criterion/$n_tasks  --beta1 $beta1 --beta2 $beta2  --epsilon 1e-8  \
 --attacked_round $attacked_round  --n_trials $n_trials --optimize_hyperparams --alpha $alpha"

  full_cmd=" $cmd
  --chkpts_dir ./chkpts/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/sgd  \
  --logs_dir ./logs/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/sgd  \
  --metadata_dir ./metadata/seeds/$seed/medical_cost/$split_criterion/$n_tasks/$batch_size/local_epochs/$n_local_steps/sgd
  --hparams_config_path ../fedklearn/configs/medical_cost/hyperparams/hp_space_attack.json \
  "
  echo $full_cmd
  eval $full_cmd
