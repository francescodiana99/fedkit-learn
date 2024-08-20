#!/bin/bash
cd ../../scripts
SCRIPT_DIR='../../scripts'

batch_size=32
local_epochs=1
learning_rate=5e-7
device="cpu"
task_name="income"
heterogeneity=0.1
num_rounds=100
seed=42
force_generation=False
download_flag=False
state="louisiana"
optimizer="sgd"
dp_flag=False
task_name="income"
split_criterion="correlation"


# Function to display help message
usage() {
    echo "Usage: $0  --batch_size BATCH_SIZE --local_epochs LOCAL_EPOCHS --learning_rate LEARNING_RATE --device DEVICE \
    --heterogeneity HETEROGENEITY --num_rounds NUM_ROUNDS --seed SEED --force_generation FORCE_GENERATION --download_flag DOWNLOAD_FLAG \
    --split_criterion SPLIT_CRITERION "
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
        --heterogeneity)
            heterogeneity="$2"
            shift 2
            ;;
        --split_criterion)
            split_criterion="$2"
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

if [ "$split_criterion" = "correlation" ]; then
    state="louisiana"
    dirs="--chkpts_dir ./$task_name/$state/heter/$heterogeneity/$batch_size/$local_epochs/ \
 --logs_dir ./logs/$task_name/$state/heter/$heterogeneity/$batch_size/$local_epochs/ \
 --metadata_dir ./metadata/$task_name/$state/heter/$heterogeneity/$batch_size/$local_epochs/ \
  --mixing_coefficient $heterogeneity"
elif [ "$split_criterion" = "state" ]; then
    state="full"
    dirs="--chkpts_dir ./$task_name/$state/$batch_size/$local_epochs/ \
    --logs_dir ./logs/$task_name/$state/$batch_size/$local_epochs/ \
    --metadata_dir ./metadata/$task_name/$state/$batch_size/$local_epochs/"
fi

if [ "$state" != "full" ]; then
    n_tasks=10
fi

if  [ "$state" = "full" ]; then
    n_tasks=51
fi




cmd="python run_simulation.py --task_name income --test_frac 0.1 --scaler standard --optimizer sgd \
--batch_size $batch_size  --learning_rate $learning_rate --device $device \
--by_epoch --local_steps $local_epochs --num_rounds $num_rounds --seed $seed \
--data_dir ./data/$task_name  --log_freq 10 --save_freq 1 \
--model_config_path ./../fedklearn/configs/$task_name/$state/models/net_config.json --split_criterion $split_criterion \
 --state $state  --n_tasks $n_tasks $dirs"


 f [ "$force_generation" = "true" ]; then
    cmd="$cmd --force_generation"
fi

if [ "$download_flag" = "true" ]; then
        cmd="$cmd --download"
fi

echo "Running command: "
echo $cmd
eval $cmd
