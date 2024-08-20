#!/bin/bash
cd ../../scripts
SCRIPT_DIR='../../scripts'

batch_size=32
local_epochs=1
learning_rate=0.005
device="cpu"
task_name="linear_medical_cost"
num_rounds=300
seed=42
force_generation=False
download_flag=False
optimizer="sgd"
split_criterion="random"
n_tasks=2

# Function to display help message
usage() {
    echo "Usage: $0  --batch_size BATCH_SIZE --local_epochs LOCAL_EPOCHS --learning_rate LEARNING_RATE --device DEVICE \
     --num_rounds NUM_ROUNDS --seed SEED --force_generation FORCE_GENERATION --download_flag DOWNLOAD_FLAG"
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
            ;;
        --force_generation)
            force_generation="true"
            shift 1
            ;;
        --download)
            download_flag="true"
            shift 1
            ;;
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
        --help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done


cmd="python run_simulation.py --task_name $task_name --test_frac 0.1 --scaler standard --optimizer sgd \
--batch_size $batch_size  --learning_rate $learning_rate --device $device \
--by_epoch --local_steps $local_epochs --num_rounds $num_rounds --seed $seed \
--data_dir ./data/$task_name  --log_freq 10 --save_freq 1 \
--model_config_path ./../fedklearn/configs/$task_name/models/linear_config.json --split_criterion $split_criterion \
  --n_tasks $n_tasks \
 --chkpts_dir ./chkpts/$task_name/$batch_size/$local_epochs/ \
 --logs_dir ./logs/$task_name/$batch_size/$local_epochs/ \
  --metadata_dir ./metadata/$task_name/$batch_size/$local_epochs/"


 if [ "$force_generation" = "true" ]; then
    cmd="$cmd --force_generation"
fi

if [ "$download_flag" = "true" ]; then
        cmd="$cmd --download"
fi

echo "Running command: "
echo $cmd
eval $cmd