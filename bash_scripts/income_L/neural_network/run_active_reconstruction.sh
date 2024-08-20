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
    alpha=0.05
else
    alpha=$3
fi

if [ -z "$4" ]; then
    beta1=0.05
else
    beta1=$4
fi

if [ -z "$5" ]; then
    beta2=0.05
else
    beta2=$5
fi

if [ -z "$6" ]; then
    n_tasks='10'
else
    n_tasks=$6
fi

if [ -z "$7" ]; then
    n_task_samples='all'
else
    n_task_samples=$7
fi

if [ -z "$8" ]; then
    state="louisiana"
else
    state=$8
fi

if [ -z "$9" ]; then
    attacked_round=99
else
    attacked_round=$9
fi

if [ -z "${10}" ]; then
    n_trials=10
else
    n_trials=${10}
fi

if [ -z "${11}" ]; then
    lr=1e-5
else
    lr=${11}
fi

if [ -z "${12}" ]; then
    seed=0
else
    seed=${12}
fi

if [ -z "${13}" ]; then
    num_rounds=0
else
    num_rounds=${13}
fi

if [ -z "${14}" ]; then
    mix=10
else
    mix=${14}
fi

if [ -z "${15}" ]; then
    device="cpu"
else
    device=${15}
fi

cmd="python run_active_simulation.py --task_name income --test_frac 0.1 --scaler standard --optimizer sgd --alpha $alpha \
--momentum 0.0  --weight_decay 0.0 --batch_size $batch_size --local_steps $n_local_steps --by_epoch --device $device \
  --log_freq 5 --save_freq 1 --num_rounds $num_rounds --seed $seed --learning_rate $lr \
--model_config_path ../fedklearn/configs/income/$state/models/net_config.json \
 --n_tasks $n_tasks --beta1 $beta1 --beta2 $beta2  --epsilon 1e-8 --attacked_round $attacked_round \
 --n_trials $n_trials  --optimize_hyperparams"

  mix_scaled=$(printf "%.0f" $(echo "$mix * 100" | bc -l))
  echo "Running simulation with mix coefficient: $mix"
  full_cmd=" $cmd \
  --data_dir ./data/seeds/$seed/income/tasks/correlation/$state/$mix_scaled/$n_tasks/$n_task_samples \
  --chkpts_dir ./chkpts/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_task_samples/$n_local_steps/sgd \
  --logs_dir ./logs/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/sgd/active \
  --metadata_dir ./metadata/seeds/$seed/income/$state/mixed/$mix_scaled/$n_tasks/$batch_size/$n_local_steps/sgd \
  --hparams_config_path ../fedklearn/configs/income/$state/hyperparams/hp_space_attack.json
  "
  echo $full_cmd
  eval $full_cmd


