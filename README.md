# FedKit-Learn

FedKit-Learn is a Python package that provides a comprehensive set of tools for
implementing and simulating federated learning scenarios.
This package is designed to assist researchers, developers, and practitioners in
exploring federated learning, federated attack simulations, and working with federated datasets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Paper Experiments](#paper-experiments)
- [Additional Experiments](#additional-experiments)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Federated learning is an emerging approach to machine learning where a model is trained
across decentralized devices or servers holding local data samples, without exchanging them.
FedKit-Learn aims to simplify the exploration and implementation of federated learning
scenarios by providing a set of tools and utilities.

## Features

* **Federated Learning Simulator**: Simulate federated learning scenarios with customizable parameters, algorithms, and communication strategies.

* **Federated Attack Simulator**: Test the robustness of your federated learning system against Attribute Inference Attacks.

* **Federated Datasets**: Access federated datasets or create your own for training and evaluation.

## Installation

Install using `setup.py`
```bash
pip install .
```
This will install the project along its dependencies.

## Usage
We provide code to simulate federated training of machine learning models (`federated/`), as well as code to simulate
machine learning attacks (`attacks/`). Moreover, we implement standard federated learning datasets in `datasets/`.

## Paper Experiments
For experiments on toy dataset, you can check [Toy Experiments](toy_experiments.ipynb).

In `bash_scripts/`, we provide scripts to simulate the experiments presented in the paper.
To replicate the results in Table 1, navigate to `bash_scripts/` and launch the following commands:

### Income-L
```bash
./run_experiments_income_l.sh
```

### Income-A
```bash
./run_experiments_income_a.sh
```

### Medical
```bash
./run_experiments_medical.sh
```


## Additional Experiments
To run additional experiments, it is possible to use and customize the following scripts, in `bash_scripts/`
### Simulate Federated Learning

These script simulate a Federated Learning simulation, and generate the following files\directories:
* data folder
* chkpts folder: models exchanged between the clients and the server
* metadata folder: metadata associated to the simulation

Flag `--download` has to be set to download the data, and  `--force_generation` to force task data splitting

#### Income-L
To launch Federated Learning simulation for a neural network model, navigate to `bash_scripts/income`, and run the following script:
```bash
sh run_simulation.sh  --batch_size 32 --local_epochs 1 --learning_rate 5e-7 --device cuda --heterogeneity 0.1 --num_rounds 100 --seed 42 --split_criterion "correlation" --force_generation  --download
```
This will generate the same setting presented in Table 1.
The simulation will create data folder, chkpts folder and metadata folder in the following locations:
* data folder: `./data/income`
* chkpts folder: `./chkpts/income/louisiana/$batch_size/$local_epochs/`
* metadata folder: `./metadata/income/louisiana/$batch_size/$local_epochs/`

To run experiments using a linear model, launch:
```bash
sh run_simulation_linear.sh --batch_size 32 --local_epochs 1 --learning_rate 5e-3 --device cuda ---num_rounds 300 --seed 42 --split_criterion "random" --force_generation  --download
```
This will generate the same setting presented in Table 2.
Folder will be located in:
* data folder: `./data/linear_income`
* chkpts folder: `./chkpts/linear_income/louisiana/$batch_size/$local_epochs/`
* metadata folder: `./metadata/linear_income/louisiana/$batch_size/$local_epochs/`


#### Income-A

To launch Federated Learning simulation for a neural network model, navigate to `bash_scripts/income`, and run the following script:
```bash
sh run_simulation.sh  --batch_size 32 --local_epochs 1 --learning_rate 1e-6 --device cuda --heterogeneity 0.1 --num_rounds 100 --seed 42 --split_criterion "state" --force_generation  --download
```

To run experiments using a linear model, launch:
```bash
sh run_simulation_linear.sh --batch_size 32 --local_epochs 1 --learning_rate 5e-3 --device cuda ---num_rounds 300 --seed 42 --split_criterion "state" --force_generation  --download
```

#### Medical

To launch Federated Learning simulation for a neural network model, navigate to `bash_scripts/medical_cost`, and run the following script:
```bash
sh run_simulation.sh  --batch_size 32 --local_epochs 1 --learning_rate 2e-6 --device cuda  --num_rounds 100 --seed 42  --force_generation  --download
```

To run experiments using a linear model, launch:
```bash
sh run_simulation_linear.sh --batch_size 32 --local_epochs 1 --learning_rate 5e-3 --device cuda ---num_rounds 300 --seed 42 --force_generation  --download
```

### Gradient Based Attacks
After simulating federated learning, and generating the metadata files, you can execute state-of-the-art gradient based attack.

#### Attribute Inference Attack

To execute AIA attacks, navigate to examples directory (`bash_scripts/`), and execute
the Python script `run_aia.sh`. This script will generate three results file: the first one, whose name is given in `--results_path`, that by default is `aia_{LR}_{ROUND_FRAC}.json`,  is a dictionary of the form `{"score": SCORE, "n_samples": N_SAMPLES}`. The second, store the cosine dissimilarity accumulated over the optimization process in the form `{"score": SCORE, "n_samples": N_SAMPLES}`. Its name will be the same as the first file, but with the appendix `_cos_sim`. Finally, there will be a third JSON file, storing the history of all the trials in the form `{"LR":{"ROUND_FRAC": (AVG SCORE, AVG COS DIS)}}`, where `AVG COS DIS` indicates the average cosine dissimilarity loss. Note that while the first two files keep information for all the client, the latter keeps track only of averge values.

#### Income-L
To run a single attack trial, with a specific learning rate and a specific fraction of rounds to consider, launch:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 10000 --device cpu --heterogeneity 0.1 --split_criterion correlation --frac 0.10 --task_name income
```
To run multiple optimization trial, replicating our effort in the paper, you can run:
```bash
run_all_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --device cpu --heterogeneity 0.1 --split_criterion correlation  --task_name income
```

To run the experiments on linear models, use:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 10000 --device cpu --heterogeneity 0.1 --split_criterion random --frac 0.10 --task_name linear_income
```
```bash
run_all_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --device cpu --heterogeneity 0.1 --split_criterion random  --task_name linear_income
```

#### Income-A
To run a single attack trial, with a specific learning rate and a specific fraction of rounds to consider, launch:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 10000 --device cpu --heterogeneity 0.1 --split_criterion state --frac 0.10 --task_name income
```
To run multiple optimization trial, replicating our effort in the paper, you can run:
```bash
run_all_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --device cpu --heterogeneity 0.1 --split_criterion state  --task_name income
```

To run the experiments on linear models, use:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 10000 --device cpu --heterogeneity 0.1 --split_criterion state --frac 0.10 --task_name linear_income
```
```bash
run_all_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --device cpu --heterogeneity 0.1 --split_criterion state  --task_name linear_income
```

#### Medical

To run a single attack trial, with a specific learning rate and a specific fraction of rounds to consider, launch:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 10000 --device cpu --heterogeneity 0.1 --split_criterion random --frac 0.10 --task_name medical_cost
```
To run multiple optimization trial, replicating our effort in the paper, you can run:
```bash
run_all_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --device cpu --heterogeneity 0.1 --split_criterion strandomate  --task_name medical_cost
```

To run the experiments on linear models, use:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 10000 --device cpu --heterogeneity 0.1 --split_criterion random --frac 0.10 --task_name linear_medical_cost
```
```bash
run_all_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --device cpu --heterogeneity 0.1 --split_criterion random  --task_name linear_medical_cost
```

### Isolation
Simulate an active adversary isolating a single client. To  execute the isolation, navigate to examples directory (`bash_scripts/income`) and execute the script `run_isolation.sh" This will generate the following metadata files/directories:
* `scripts/isolated`: directory containing the isolated models.
* `isolated_trajectories_{ROUND}.json`: dictionary containing models' trajectories of the isolated models, when the attack starts from ROUND.
* `isolated_{ROUND}.json` dictionary containing last round models' of the attack starting from ROUND.

#### Income-L

This script also takes the option `--attacked_task`, which allows to execute the isolation process only on one client, to reduce the execution time.
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 5e-7 --device cpu --heterogeneity 0.1 --split_criterion correlation --attacked_round 99 --task_name income --attacked_task 1
```

To isolate the linear model, run:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 5e-3 --device cpu --split_criterion random --attacked_round 99 --task_name linear_income --attacked_task 1
```

#### Income-A

```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 1e-6 --device cpu  --split_criterion state --attacked_round 99 --task_name income --attacked_task 1
```

To isolate the linear model, run:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 5e-3 --device cpu  --split_criterion state --attacked_round 99 --task_name linear_income --attacked_task 1
```

#### Medical

```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 2e-6 --device cpu    --attacked_round 99 --task_name medical_cost --attacked_task 1
```

To isolate the linear model, run:
```bash
run_aia.sh --num_rounds 100 --seed 42 -batch_size 32  --local_epochs 1 --learnin_rate 5e-3 --device cpu   --attacked_round 99 --task_name linear_medical_cost --attacked_task 1
```

### Active Gradient Based Attribute Inference Attack


### Execute Local Model Reconstruction Attack

To execute LMRA attack, navigate to examples directory (`scripts/`), and execute
the Python script `run_lmra.py`. The script generates the following files\directories:
* reconstructed models folder: contains a folder for each client. Each folder stores the reconstructed models of each
client at different iterations.
* reconstruction metadata files:
  * `reconstructed.json`:
  * `trajectory.json`:
* results file: The results are saved in a JSON file, storing a list of the same size as the
number of clients. Each element is a dictionary of the form `{"score": SCORE, "n_samples": N_SAMPLES}`.

The script also takes an option `--use_oracle`. If selected, the local model reconstruction attack uses
an oracle to compute the gradients instead of estimating them.
Note that using the oracle is almost equivalent to using SGD with the full gradient.

#### Adult Dataset

```bash
cd scripts/

python run_lmra.py \
  --task_name adult \
  --data_dir ./data/adult \
  --split train \
  --metadata_dir ./metadata/adult/ \
  --hidden_layers 1024 \
  --use_oracle \
  --optimizer sgd \
  --estimation_learning_rate 0.01 \
  --reconstruction_learning_rate 0.03 \
  --momentum 0.0 \
  --weight_decay 0.0 \
  --batch_size 1024 \
  --num_rounds 100
  --device cpu \
  --logs_dir ./logs/adult \
  --log_freq 1 \
  --reconstructed_models_dir ./reconstructed_models/adult \
  --save_freq 1 \
  --results_path ./results/toy_classification/lmra.json \
  --seed 42 \
  --debug \
  -v
```

#### Toy Classification Dataset

```bash
cd scripts/

python run_lmra.py \
  --task_name toy_classification \
  --data_dir ./data/toy_classification \
  --split train \
  --metadata_dir ./metadata/toy_classification/ \
  --hidden_layers 1024 \
  --use_oracle \
  --optimizer sgd \
  --estimation_learning_rate 0.01 \
  --reconstruction_learning_rate 0.03 \
  --momentum 0.0 \
  --weight_decay 0.0 \
  --batch_size 1024 \
  --num_rounds 100 \
  --device cpu \
  --logs_dir ./logs/toy_classification \
  --log_freq 1 \
  --reconstructed_models_dir ./reconstructed_models/toy_classification \
  --save_freq 1 \
  --results_path ./results/toy_classification/lmra.json \
  --seed 42 \
  --debug \
  -v
```

### Evaluate Local Model Reconstruction Attack

Once LMRA is executed, you can evaluate the performance of the reconstructed model when used in other attacks.

#### Adult Dataset

```bash
cd scripts/

python evaluate_reconstruction.py \
  --task_name adult \
  --data_dir ../data/adult \
  --split train \
  --models_metadata_path ./metadata/adult/reconstructed.json \
  --reference_models_metadata_path ./metadata/adult/local.json \
  --models_config_metadata_path ./metadata/adult/model_config.json \
  --sensitive_attribute sex_Male \
  --sensitive_attribute_type binary \
  --results_dir ./results/adult/reconstructed \
  --seed 42 \
  -v
```

#### Synthetic Dataset

```bash
cd scripts/

python evaluate_reconstruction.py \
  --task_name toy_classification  \
  --data_dir ./data/toy_classification \
  --split train \
  --models_metadata_path ./metadata/toy_classification/reconstructed.json \
  --reference_models_metadata_path ./metadata/toy_classification/local.json \
  --models_config_metadata_path ./metadata/toy_classification/model_config.json \
  --results_dir ./results/toy_classification/reconstructed \
  --seed 42 \
  -v
```
**Evaluate Trajectory.** One can also evaluate each reconstructed model

#### Adult Dataset (Evaluate Trajectory)
```bash
cd scripts/

python evaluate_trajectory.py \
  --task_name adult \
  --data_dir ../data/adult \
  --split train \
  --models_metadata_path ./metadata/adult/trajectory.json \
  --reference_models_metadata_path ./metadata/adult/local.json \
  --models_config_metadata_path ./metadata/adult/model_config.json \
  --sensitive_attribute sex_Male \
  --sensitive_attribute_type binary \
  --results_dir ./results/adult/reconstructed \
  --plots_dir ../plots/adult/reconstructed \
  --seed 42 \
  -v
```


## Contributing
We welcome contributions! To contribute to FedKit-Learn, please follow the guidelines
outlined in `CONTRIBUTING.md`.
We appreciate bug reports, feature requests, and pull requests.

## License
FedKit-Learn is released under the Apache License 2.0. Feel free to use, modify,
and distribute this package in accordance with the terms of the Apache License 2.0.
