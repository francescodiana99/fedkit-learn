#!/bin/bash
cd ../../../scripts

if [ -z "$1" ]; then
    batch_size=32
else
    batch_size=$1
fi

if [ -z "$2" ]; then
    learning_rate=0.01
else
    learning_rate=$2
fi

if [ -z "$3" ]; then
    seed="sgd"
else
    seed=$3
fi

if [ -z "$4" ]; then
    attacked_task="sgd"
else
    attacked_task=$4
fi

if [ -z "$5" ]; then
    device="cpu"
else
    device=$5
fi

attacked_rounds=(299)
state='louisiana'
n_tasks=10
local_epochs=1
n_local_steps=1
optimizer='sgd'


base_cmd="python run_isolation.py \
  --by_epoch \
  --data_dir ./data/seeds/42/linear_income/tasks/random/$state/$n_tasks/all \
  --task_name linear_income \
  --split train \
  --metadata_dir ./metadata/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer \
  --optimizer sgd \
  --momentum 0. \
  --weight_decay 0.0 \
  --batch_size $batch_size \
  --num_epochs 50 \
  --device $device \
  --logs_dir ./logs/seeds/$seed/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer/isolated \
  --log_freq 1 \
  --save_freq 1 \
  --seed $seed \
  --attacked_task $attacked_task "

for round in "${attacked_rounds[@]}"; do
  # Construct command with current hyperparameters
  cmd="$base_cmd --learning_rate $learning_rate --attacked_round $round \
  --isolated_models_dir ./isolated/linear_income/$state/random/$n_tasks/$batch_size/$n_local_steps/$optimizer/$round"
echo "Running command: $cmd"
  # Execute the command
  $cmd
done

