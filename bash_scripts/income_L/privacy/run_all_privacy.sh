#!bin/bash

device=$1
force_generation=$2
download=$3


echo "Running simulation using DP for Income-L"

sh run_simulation.sh 32 1 5e-7 10 42 0.1 1 3e6 $device $force_generation $download

echo "Running passive Gradient Based Attack"

cd ../bash_scripts/income_L/neural_network

sh run_aia_passive_gb.sh 32 1 10 louisiana sgd 42 1 3e6 $device

echo "Passive Gradient Based Attack is complete"

cd ../bash_scripts/income_L/neural_network

echo "Running isolation"

sh run_isolation.sh 32 1 10 5e-7 42  99  1 3e6 $device

echo "Isolation is complete"

cd ../bash_scripts/income_L/neural_network

echo "Running active gradient based attack"

sh run_aia_active_gb.sh 32 1 10 louisiana sgd 42  1 3e6 99 $device

echo "Active gradient based attack is complete"

cd ../bash_scripts/income_L/neural_network

echo "Running model based attack"

sh run_aia_mb.sh 32 1 99 42 49 10 1 3e6 $device




