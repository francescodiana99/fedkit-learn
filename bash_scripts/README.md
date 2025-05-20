# How to run the experiments
This folder contains a collection of bash scripts for reproducing the experiments presented in our paper. Each subfolder corresponds to a specific dataset and includes the necessary bash scripts. The experiments are organized into the following categories:
- **neural_network**– Scripts to reproduce the experiments shown in **Table 1** of the main paper.
- **linear_regression** – Scripts to reproduce the experiments in **Table 2** (Appendix C.1).
- **privacy** – Scripts to reproduce the experiments in **Table 3** (Appendix C.4).

## Scripts
Each bash script runs a specific experiment or attack simulation. You can customize these scripts as needed:

- `run_simulation.sh`: Runs the federated learning simulation with an honest server. This must be executed before any attack simulations.  诚实的服务器

- `run_aia_passive_gb.sh`: Simulates the gradient-based attack from [Lyu et al.](https://arxiv.org/abs/2108.06910). 基于梯度的攻击

- `run isolation.sh`: Simulates a malicious server that isolates clients by always returning the model it receives.
- `run_aia_active_gb.sh`: Runs the attack from [Lyu et al.](https://arxiv.org/abs/2108.06910) on clients that have been isolated.

- `run_local_optimal_model_search.sh`: Performs a hyperparameter search to find the empirical optimal minimizer of the training loss (i.e., the Model-with-Oracle setting in our experiments).

- `run_active_reconstruction.sh`: Simulates the active reconstruction attack described in **Algorithm 3** of the paper.

- `run_aia_mb.sh`: Evaluates the model-based Attribute Inference Attack. 模型属性推理攻击

- `run_table_1_experiments.sh`: Reproduces **Table 1** results using the setup described in **Appendix B.5**.

- run_linear_reconstruction.sh: Executes **Algorithm 2** to simulate a passive adversary reconstructing the optimal local model in a least-squares regression scenario.

- `run_table_2_experiments.sh`: Reproduces **Table 2** results using the setup in **Appendix C.1**.
  
- `run_table_3_experiments.sh`: Reproduces **Table 3** results using the setup in **Appendix C.4**.

## Customization
It is possible to define and customize the neural network model by modifying the `net_config.json` file in `fedklearn\configs\<dataset_name>\models` and by adding your own model in `fedklearn\models`.
To modify the hyperparameter search space used in Algorithm 3, update the configuration file `fedklearn\configs\<dataset_name>\hyperparameters\hp_space_attack.json`.

