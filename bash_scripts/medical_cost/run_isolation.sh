cd ../../scripts
SCRIPT_DIR='../../scripts'

batch_size=32
local_epochs=1
device="cpu"
task_name="medical_cost"
num_rounds=100
seed=42
split_criterion="random"
attacked_round=99
learning_rate=2e-6


# Function to display help message
usage() {
    echo "Usage: $0  --batch_size BATCH_SIZE --local_epochs LOCAL_EPOCHS --learning_rate LEARNING_RATE --device DEVICE \
     --num_rounds NUM_ROUNDS --seed SEED --split_criterion SPLIT_CRITERION --task_name TASK_NAME -attacked_round --attacked_task ATTACKED_TASK"
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
        --split_criterion)
            split_criterion="$2"
            shift 2
            ;;
        --attacked_round)
            attacked_round="$2"
            shift 2
            ;;
        --task_name)
            task_name="$2"
            shift 2
            ;;
        --attacked_task)
            attacked_task="$2"
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

        else
            dirs="--logs_dir ./logs/$task_name/$batch_size/$local_epochs/isolated/$attacked_round \
                --metadata_dir ./metadata/$task_name/$batch_size/$local_epochs/ \
                --data_dir ./data/$task_name/tasks/$split_criterion/$n_tasks/all \
                --isolated_models_dir ./isolated/$task_name/$batch_size/$local_epochs/$attacked_round"


        fi



base_cmd="python run_isolation.py \
  --by_epoch \
  --task_name income \
  --split train \
  --optimizer sgd \
  --momentum 0. \
  --weight_decay 0.0 \
  --batch_size $batch_size \
  --num_epochs $num_rounds \
  --device cuda \
  --log_freq 1 \
  --save_freq 1 \
  --seed $seed \
  --attacked_round $attacked_round"

if [ "$attacked_task" ]; then
    cmd="$base_cmd --attacked_task $attacked_task"
else
    cmd="$base_cmd"
fi

echo $cmd
eval $cmd