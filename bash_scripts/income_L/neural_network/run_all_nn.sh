#!bin/bash

device=$1
force_flag=$2
download=$3



echo "Running simulation for Income-L"
sh run_simulation.sh 32 1 5e-7 100 42 0.1 $force_flag $download

echo "Running passive Gradient Based Attack"

cd ../bash_scripts/income_L/neural_network

sh run_aia_passive_gb.sh 32 1 10 louisiana sgd 42

echo Gradient Based Attack is complete

cd ../bash_scripts/income_L/neural_network

echo "Running active attack"

sh run__active_reconstruction 32 1 5e-7 1 99 99 10 all louisiana  99 50 5e-7 42 50 10 $device

echo Active reconstruction is complete

cd ../bash_scripts/income_L/neural_network

echo "Running isolation"

sh run_isolation.sh 32 1 10 5e-7 42 $device

echo Isolation is complete

cd ../bash_scripts/income_L/neural_network

echo "Running Gradient based active attack"


sh run_aia_active_gb.sh 32 1 10 louisiana sgd 42 99 $device

echo Gradient based active attack is complete

cd ../bash_scripts/income_L/neural_network

echo "Running Model Based Attack"

sh run_aia_model_based.sh 32 1 99 42 49 10 $device


