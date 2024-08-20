#!/bin/bash
cd ../../../scripts


if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    attacked_round=99
else
    attacked_round=$2
fi

if [ -z "${3}" ]; then
    n_trials=10
else
    n_trials=${3}
fi

if [ -z "${4}" ]; then
    lr=1e-5
else
    lr=${4}
fi

if [ -z "${5}" ]; then
    seed=0
else
    seed=${5}
fi

if [ -z "${6}" ]; then
    attacked_task=0
else
    attacked_task=${6}
fi

if [ -z "${7}" ]; then
    device="cuda"
else
    device=${7}
fi

num_rounds=50
n_local_steps=1
state="louisiana"
beta1=0.99
beta2=0.999
alpha=1
n_task_samples='all'
n_tasks=10
split_criterion='random'
optimizer="sgd"


cmd="python run_active_simulation.py --task_name linear_income --test_frac 0.1 --scaler standard --optimizer sgd --alpha $alpha \
--momentum 0.0  --weight_decay 0.0 --batch_size $batch_size --local_steps $n_local_steps --by_epoch --device $device \
  --log_freq 10 --save_freq 1 --num_rounds $num_rounds --seed $seed --learning_rate $lr \
--model_config_path ../fedklearn/configs/income/$state/models/config_linear.json \
 --n_tasks $n_tasks --beta1 $beta1 --beta2 $beta2  --epsilon 1e-8 --attacked_round $attacked_round \
 --n_trials $n_trials  --optimize_hyperparams --attacked_task $attacked_task"

full_cmd=" $cmd \
  --data_dir ./data/seeds/42/linear_income/tasks/random/$state/$n_tasks/all \
  --chkpts_dir ./chkpts/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer \
  --logs_dir ./logs/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer/active \
  --metadata_dir ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer \
  --hparams_config_path ../fedklearn/configs/income/$state/hyperparameters/hp_space_attack.json
  "
  echo $full_cmd
  eval $full_cmd

# Check if the Python command was successful
if [ $? -ne 0 ]; then
  echo "Failed to run the script"
  exit 0
fi

echo "Script executed successfully"
