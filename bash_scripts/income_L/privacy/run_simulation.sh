
#!/bin/bash
cd ../../../scripts
force_flag=false

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
    lr=0.05
else
    lr=$3
fi

if [ -z "$4" ]; then
    num_rounds=100
else
    num_rounds=$4
fi

if [ -z "$5" ]; then
    seed=0
else
    seed=$5
fi

if [ -z "$6" ]; then
    mix=0.1
else
    mix=$6
fi

if [ -z "$7" ]; then
    epsilon=1
else
    epsilon=$7
fi

if [ -z "$8" ]; then
    clip=1
else
    clip=$8
fi

if [ -z "$9" ]; then
    device="cpu"
else
    device=$9
fi



state="louisiana"
optimizer="sgd"
n_trials=50
attacked_task=3
active_rounds=50
delta=1e-5


cmd="python run_simulation.py --task_name income --test_frac 0.1 --scaler standard --optimizer $optimizer --learning_rate $lr \
--momentum 0.0  --weight_decay 0.0 --batch_size $batch_size --local_steps $n_local_steps --by_epoch --device $device \
--data_dir ./data/seeds/$seed/income  --log_freq 5 --save_freq 1 --num_rounds $num_rounds --seed $seed \
--model_config_path ../fedklearn/configs/income/$state/models/config_1.json --split_criterion correlation \
--n_tasks 10 --state $state --dp_epsilon $epsilon --dp_delta $delta --clip_norm $clip  --use_dp \
--dp_delta 1e-5 --dp_epsilon $epsilon --num_active_rounds $active_rounds \
--attacked_task $attacked_task  --n_trials $n_trials "

  mix_scaled=$(printf "%.0f" $(echo "$mix * 100" | bc -l))
  echo "Running simulation with mix coefficient: $mix"
  full_cmd=" $cmd --mixing_coefficient $mix \
  --chkpts_dir ./chkpts/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_scaled/10/$batch_size/$n_local_steps/$optimizer \
  --logs_dir ./logs/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_scaled/10/$batch_size/$n_local_steps/$optimizer \
  --metadata_dir ./metadata/seeds/$seed/privacy/$epsilon/$clip/income/$state/mixed/$mix_scaled/10/$batch_size/$n_local_steps/$optimizer \
  --hparams_config_path ../fedklearn/configs/income/$state/hyperparams/hp_space_attack.json"
  echo $full_cmd
  eval $full_cmd