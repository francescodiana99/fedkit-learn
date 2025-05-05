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
sh run_simulation.sh 32 1 5e-3 100 $seed  $device "--force_generation --download"
echo "Simulation complete. Running passive Gradient Based AIA"
cd $scripts_dir

sh run_aia_passive_gb.sh $seed $device 
echo "Successfully completed passive Gradient Based AIA"
cd $scripts_dir

echo "Running malicious server reconstruction attack"
sh run_active_reconstruction.sh 99 50 $seed $device  
echo "Active reconstruction is complete"
cd $scripts_dir

echo "Running clients isolation"
sh run_isolation.sh $seed $device 
echo "Clients isolation is complete"
cd $scripts_dir

echo "Running Gradient based active AIA"
sh run_aia_active_gb.sh $seed $device 
echo "Gradient based active AIA is complete"
cd $scripts_dir

echo "Running Optimal Local Model search"
sh run_local_optimal_model_search.sh 200 50 $seed $device 
echo "Optimal Local Model search is complete"
cd $scripts_dir

echo "Running Passive Model Based Attack"
sh run_linear_reconstruction.sh  42 10000000 $device
cd $scripts_dir

echo "Running Active Model Based Attack"
sh run_aia_mb.sh  $device $seed 99 
echo "Model Based Attack is complete"
cd $scripts_dir

echo "Simulation completed"


