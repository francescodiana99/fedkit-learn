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
To replicate the results on Income-L, navigate to `bash_scripts/Income-L` and launch the following commands:
From `Income-L/neural_network` launch
### Income-L
```bash
sh run_all_nn.sh  cuda
```
to launch all the script to perform the experiments presented in the main Table of the paper.
It is also possible to launch the script using the CPU, by calling:
```bash
sh run_all_nn.sh  cpu
```

For the linear approach proposed in Appendix C.1, you can launch:
```bash
sh run_all_linear.sh  cuda force_generation download
```
Adding `force_generation`, forces to compute each clients' dataset, while `download` forces to download the data.

```bash
sh run_all_linear.sh  cuda force_generation download
```
Finally, for the experiments on the effect of Differential Privacy, you can launch:
```bash
sh run_all_privacy.sh  cuda force_generation download
```

Note that each of these script calls a series of scripts that allow to customize and perform each operation separately.

## Additional Experiments
To run additional experiments on Income-A and Medical, it is possible to use and customize all the scripts in `bash_scripts/`. Here we show some example. Note thaat the parameters have to be passed in the same order as reported in the examples.

### Income-A

#### Simulation
To launch Federated Learning simulation for a neural network model, navigate to `bash_scripts/income_A/neural_network`, and run the following script:
```bash
sh run_simulation.sh  BATCH_SIZE LOCAL_EPOCHS LR N_ROUNDS SEED DEVICE force_flag download
```
Our configuration
```bash
sh run_simulation.sh  32 1 1e-6 1 42 force_flag download
```

#### Passive Gradient Based Attack

This attack will generate a dictionary called `all_aia.json` containing the average performance of all the tested settings. It also will create for each combination of learning rate and fraction of observed rounds, two specific json files: `aia_{lr}_{frac}.json`, containing the scores for each client, and `aia_{lr}_{frac}_cos_dis.json`, containing the cosine dissimilarity loss correspondent to each client.

```bash
sh run_aia_passive_gb.sh  BATCH_SIZE LOCAL_EPOCHS OPTIMIZER SEED DEVICE
```
Our configuration:
```bash
sh run_aia_passive_gb.sh  32 1 sgd 42 cuda
```
#### Active Gradient Based Attack

 This attacks relies on two steps:
 * Isolate a client and train for additional 50 rounds.
 * Perform the gradient based attack

The first script train the attacked client, saving the metadata in `./metadata` folder, while the second produces the same outputs of the passive attack.

```bash
sh run_isolation.sh  BATCH_SIZE LOCAL_EPOCHS LR  DEVICE SEED
```
Our configuration:
```bash
sh run_isolation.sh  32 1 1e-6  cuda 42
```

```bash
sh run_aia_active_gb.sh  BATCH_SIZE LOCAL_EPOCHS  SEED  ATTACKED_ROUND DEVICE
```
Our configuration:
```bash
sh run_aia_active_gb.sh  32 1  42 99 cuda
```

#### Active Reconstruction

 Then, we simulate the active adversary scenario, proposed in Algorithm 4. This script will savve models' trajectories in the `./metadata` folder. Note that here, when running the optimization, ALPHA, BETA1 and BETA2 are placeholders

```bash
sh run_active_reconstruction.sh  BATCH_SIZE LOCAL_EPOCHS  ALPHA BETA1 Beta2 N_TASKS N_TASK_SAMPLES STATE ATTACKED_ROUNDS N_TRIALS LR SEED   DEVICE
```
Our configuration:
```bash
sh run_aia_active_gb.sh  32 1  0 0 0 51 39133 full 99 50 1e-6 42  cuda
```

#### Model-w-O Generation
To generate the empirical optimal model, run the following script, that will create the metadata in `./metadata` folder :

```bash
sh run_active_reconstruction.sh  BATCH_SIZE N_ROUNDS TRIALS SEED DEVICE
```
Our configuration:
```bash
sh run_active_reconstruction.sh  32 200 50 42 cuda
```


#### Evaluate the reconstruction

Finally to evaluate our model-based attack, launch:

```bash
sh run_aia_mb.sh  BATCH_SIZE LOCAL_EPOCHS ATTACKED_ROUND SEED   DEVICE
```
Our configuration:
```bash
sh run_active_reconstruction.sh  32 1 99 42 cuda
```
This wil evaluate the attack after 1, 10 and 50 Active Rounds


### Medical

Additional experiments have been performed on Medical dataset. To perform them, refer to the `bash_scripts/medical_cost` directory.