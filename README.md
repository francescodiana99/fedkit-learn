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
- [Contributing](#contributing)
- [License](#license)

## Introduction 

Federated learning is an emerging approach to machine learning where a model is trained
across decentralized devices or servers holding local data samples, without exchanging them.
FedKit-Learn aims to simplify the exploration and implementation of federated learning 
scenarios by providing a set of tools and utilities.

## Features 

* **Federated Learning Simulator**: Simulate federated learning scenarios with customizable parameters, algorithms, and communication strategies.

* **Federated Attack Simulator**: Test the robustness of your federated learning system against various attacks and adversarial scenarios.

* **Federated Datasets**: Access federated datasets or create your own for training and evaluation.

* **Customization**: Easily customize federated learning algorithms, models, and communication strategies to suit your specific requirements.

## Installation

To clone the repository, run
```bash
git clone https://gitlab.inria.fr/omarfoq/fedkit-learn.git
cd fedkit-learn
```
Install using `setup.py`
```bash
pip install .
```
This will install the project along its dependencies. 

## Usage 
We provide code to simulate federated training of machine learning models (`federated/`), as well as code to simulate 
machine learning attacks (`attacks/`). Moreover, we implement standard federated learning datasets in `datasets/`. 

## Paper Experiments
In `scripts/`, we provide scripts to simulate federated learning (`scripts/run_simulation.py`), 
as long as scripts to simulate federated attacks. The list of supported attacks includes:
Attribute Inference Attack (`scripts/run_aia.py`), Source Inference Attack (`scripts/run_sia.py`), 
Sample Reconstruction Attack (`scripts/run_sra.py`), and Local Model Reconstruction Attack (`scripts/run_lmra.py`).
Additionally, we provide two scripts to evaluate reconstructed models (`scripts/evaluate_reconstructed_models.py`
and `scripts/evaluate_oracle_models.py`).

### Simulate Federated Learning
To simulate federated learning, navigate to examples directory (`scripts/`), and execute 
the Python script `run_simulation.py`. The script generates the following files\directories:
* data folder
* local models folder: local optimal models associated to each client; 
i.e., trained on the client local dataset with no collaboration
* chkpts folder: models exchanged between the clients and the server
* metadata folder: it contains three JSON files
  * `federated.json`:
  * `last.json`:
  * `server.json`: 
  * `local.json`: 

#### Adult Dataset 

```bash
cd scripts/

python run_simulation.py \
    --task_name adult \
    --test_frac none \
    --scaler standard \
    --compute_local_models \
    --optimizer sgd \
    --learning_rate 0.03 \
    --momentum 0.0 \
    --weight_decay 0.0 \
    --batch_size 1024 \
    --local_steps 1 \
    --by_epoch \
    --device cpu \
    --data_dir ./data/adult \
    --local_models_dir ./local_models/adult \
    --chkpts_dir ./chkpts/adult \
    --logs_dir ./logs/adult/training \
    --metadata_dir ./metadata/adult \
    --log_freq 10 \
    --save_freq 1 \
    --num_rounds 100 \
    --seed 42 
```

#### Toy Classification Dataset

```bash
cd scripts/

python run_simulation.py \
    --task_name toy_classification \
    --n_tasks 2 \
    --n_train_samples 10 \
    --n_test_samples 1000 \
    --n_numerical_features 0 \
    --n_binary_features 1 \
    --sensitive_attribute_type binary \
    --sensitive_attribute_weight  0.5 \
    --noise_level 0.0 \
    --compute_local_models \
    --optimizer sgd \
    --learning_rate 0.1 \
    --momentum 0.0 \
    --weight_decay 0.0 \
    --batch_size 1024 \
    --local_steps 1 \
    --by_epoch \
    --device cpu \
    --data_dir ./data/toy_classification \
    --local_models_dir local_models/toy_classification \
    --chkpts_dir ./chkpts/toy_classification \
    --logs_dir ./logs/toy_classification/training \
    --metadata_dir ./metadata/toy_classification \
    --log_freq 10 \
    --save_freq 1 \
    --num_rounds 200 \
    --seed 42 \
```
### State-of-the-art Attacks
After simulating federated learning, and generating the metadata files, you can execute state-of-the-art attacks.  

#### Attribute Inference Attack

To execute AIA attack, navigate to examples directory (`scripts/`), and execute 
the Python script `run_aia.py`. The results are saved in a JSON file, storing a list of the same size as the 
number of clients. Each element is a dictionary of the form `{"score": SCORE, "n_samples": N_SAMPLES}`. 

#### Adult Dataset

```bash
cd scripts/

python run_aia.py \
    --task_name adult \
    --keep_rounds_frac 0.0 \
    --temperature 1.0 \
    --metadata_path ./metadata/adult/federated.json \
    --data_dir ./data/toy_classification \
    --split train \
    --optimizer sgd \
    --learning_rate 1. \
    --num_rounds 200 \
    --device cpu \
    --logs_dir ./logs/adult/aia \
    --log_freq 1 \
    --results_path ./results/adult/aia.json \
    --seed 42 
```

#### Toy Classification Dataset

```bash
cd scripts/

python run_aia.py \
    --task_name toy_classification \
    --keep_rounds_frac 0.0 \
    --temperature 1.0 \
    --metadata_path ./metadata/toy_classification/federated.json \
    --data_dir ./data/toy_classification \
    --split train \
    --optimizer sgd \
    --learning_rate 1. \
    --num_rounds 200 \
    --device cpu \
    --logs_dir ./logs/toy_classification/aia \
    --log_freq 1 \
    --results_path ./results/toy_classification/aia.json \
    --seed 42 
```

#### Source Inference Attack

To execute SIA attack, navigate to examples directory (`scripts/`), and execute 
the Python script `run_sia.py`. The results are saved in a JSON file, storing a list of the same size as the 
number of clients. Each element is a dictionary of the form `{"score": SCORE, "n_samples": N_SAMPLES}`. 

#### Adult Dataset

```bash
cd scripts/

python run_aia.py \
  --task_name adult \
  --sensitive_attribute sex_Male \
  --sensitive_attribute_type binary \ 
  --metadata_dir ./metadata/adult/ \
  --data_dir ./data/adult \
  --split train \
  --batch_size 1024 \
  --device cpu \
  --results_path ./results/adult/sia.json \
  --seed 42
```

#### Toy Classification Dataset

```bash
cd scripts/

python run_aia.py \
  --task_name toy_classification \
  --metadata_dir ./metadata/toy_classification/ \
  --data_dir ./data/toy_classification \
  --split train \
  --batch_size 1024 \
  --device cpu \
  --results_path ./results/toy_classification/sia.json \
  --seed 42
```

#### Sample Reconstruction Attack
**Coming soon...**

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
  --reconstruction_learning_rate 0.3 \
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
  --reconstruction_learning_rate 0.3 \
  --momentum 0.0 \
  --weight_decay 0.0 \
  --batch_size 1024 \
  --num_rounds 100
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

## Contributing
We welcome contributions! To contribute to FedKit-Learn, please follow the guidelines 
outlined in `CONTRIBUTING.md`.
We appreciate bug reports, feature requests, and pull requests.

## License
FedKit-Learn is released under the [Apache License 2.0](LICENSE). Feel free to use, modify, 
and distribute this package in accordance with the terms of the Apache License 2.0.