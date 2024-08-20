#!bin/bash

device=$1
force_flag=$2
download=$3

echo "Running simulation for Income-L"

sh run_simulation.sh 32 5e-3 100 42 $device $force_flag $download

cd ../bash_scripts/income_L/neural_network

echo "Running passive Gradient Based Attack"

sh run_aia_passive_gb.sh 32 42 $device

echo Gradient Based Attack is complete

cd ../bash_scripts/income_L/neural_network

echo "Running isolation"

sh run_isolation.sh 32 5e-3 42 3 $device

echo Isolation is complete

cd ../bash_scripts/income_L/neural_network

echo "Running Gradient based active attack"

sh run_aia_active_gb.sh 32 1  42 3 $device

echo Gradient based active attack is complete

cd ../bash_scripts/income_L/neural_network

echo "Running Active reconstruction"

sh run_active_reconstruction.sh 32 99 50 5e-3 42 3 $device

echo Model Based Attack evaluation is complete
. start.h
cd ../bash_scripts/income_L/neural_network

sh run_aia_mb.sh 32 1 99 42 3 $device

echo Model Based Attack evaluation is complete

cd ../bash_scripts/income_L/neural_network

echo "Running linear reconstruction"

sh run_linear_reconstruction.sh 32 42 10000000 $device

