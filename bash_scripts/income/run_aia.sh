#!/bin/bash
cd ../../scripts
SCRIPT_DIR='../../scripts'

batch_size=32
local_epochs=1
learning_rate=1000
device="cpu"
heterogeneity=0.1
num_rounds=100
seed=42
state="louisiana"
task_name="income"
split_criterion="correlation"
n_tasks=10
frac=0.1

# Function to display help message
usage() {
    echo "Usage: $0  --batch_size BATCH_SIZE --local_epochs LOCAL_EPOCHS --learning_rate LEARNING_RATE --device DEVICE \
    --heterogeneity HETEROGENEITY --num_rounds NUM_ROUNDS --seed SEED  --split_criterion SPLIT_CRITERION --frac FRAC --attacked_task ATTACKED_TASK"
    exit 1
}

while [ "$#" -gt 0 ]; do  # Use single brackets here
    case $1 in

        --num_rounds)
            num_rounds="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --local_epochs)
            local_epochs="$2"
            shift 2
            ;;
        --learning_rate)
            learning_rate="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --heterogeneity)
            heterogeneity="$2"
            shift 2
            ;;
        --split_criterion)
            split_criterion="$2"
            shift 2
            ;;
        --attacked_task)
            attacked_task="$2"
            shift 2
            ;;

        --frac)
            frac="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

if [ "$state" != "full" ]; then
    n_tasks=10

elif  [ "$state" = "full" ]; then
    n_tasks=51
fi


if [ "$split_criterion" = "correlation" ]; then
    mix_perc=$(echo "$heterogeneity * 100" |bc -\l)
    state="louisiana"
    dirs="--logs_dir ./logs/$task_name/$state/heter/$heterogeneity/$batch_size/$local_epochs/aia/ \
        --metadata_path ./metadata/$task_name/$state/heter/$heterogeneity/$batch_size/$local_epochs/federated.json \
        --results_path ./results/$task_name/$state/heter/$heterogeneity/$batch_size/$local_epochs/aia_${learning_rate}_${frac}.json \
        --data_dir ./data/$task_name/tasks/$split_criterion/$state/$mix_perc/$n_tasks/all"

elif [ "$split_criterion" = "state" ]; then
    state="full"
    dirs="--logs_dir ./logs/$task_name/$state/$batch_size/$local_epochs/ \
        --metadata_path ./metadata/$task_name/$state/$batch_size/$local_epochs/federated.json \
        --data_dir ./data/$task_name/tasks/$split_criterion/full/$n_tasks/39133"
fi


cmd="python run_aia.py \
  --task_name income \
  --temperature 1.0 \
  --split train \
  --optimizer sgd \
  --num_rounds 100 \
  --device cuda \
  --log_freq 10 \
  --seed $seed \
  --sensitive_attribute_type binary --learning_rate $learning_rate --sensitive_attribute SEX --keep_first_rounds --keep_rounds_frac $frac \
  $dirs"

  if [ "$attacked_task" ]; then
    cmd="$cmd --attacked_task $attacked_task"
  echo $cmd
    eval $cmd
