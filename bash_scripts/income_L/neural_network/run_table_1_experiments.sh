#!bin/bash

scripts_dir=$(pwd)

if [ -z $1 ]; then
    device='cuda'
else
    device=$1
fi

if [ -z $2 ]; then
    seed=42
else
    seed=$2
fi

echo "Running simulation for Income-L"
sh run_simulation.sh 32 1 5e-7 100 $seed 0.1 $device "--force_generation --download"
echo "Simulation complete. Running passive Gradient Based AIA"
cd $scripts_dir

sh run_aia_passive_gb.sh $seed $device 10
echo "Successfully completed passive Gradient Based AIA"
cd $scripts_dir

echo "Running malicious server reconstruction attack"
sh run_active_reconstruction.sh 99 50 $seed $device 10 
echo "Active reconstruction is complete"
cd $scripts_dir

echo "Running clients isolation"
sh run_isolation.sh $seed $device 10
echo "Clients isolation is complete"
cd $scripts_dir

echo "Running Gradient based active AIA"
sh run_aia_active_gb.sh $seed $device 10
echo "Gradient based active AIA is complete"
cd $scripts_dir

echo "Running Optimal Local Model search"
sh run_optimal_local_model_search.sh 200 50 $seed $device 10
echo "Optimal Local Model search is complete"
cd $scripts_dir

echo "Running Model Based Attack"
sh run_aia_mb.sh  $device $seed 99 10
echo "Model Based Attack is complete"
cd $scripts_dir

echo "Simulation completed"
