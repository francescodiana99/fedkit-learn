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
- [Examples](#Examples)
- [Contributing](#Contributing)
- [License](#License)

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

## Examples
In `examples/`, we provide scripts to simulate federated learning (`examples/run_simulation.py`), 
as long as scripts to simulate federated attacks. The list of supported attacks includes:
Attribute Inference Attack (`examples/run_aia.py`), Source Inference Attack (`examples/run_sia.py`), 
Sample Reconstruction Attack (`examples/run_sra.py`), and Local Model Reconstruction Attack (`examples/run_lmra.py`). 

### Simulate Federated Learning
To simulate federated learning on Adult dataset, navigate to examples directory (`examples/`), and execute 
the Python script `examples/run_simulation.py` 
```bash
cd examples/

python run_simulation.py \
    --task_name adult \
    --test_frac none \
    --scaler standard \
    --optimizer sgd \
    --learning_rate 0.03 \
    --momentum 0.0 \
    --weight_decay 0.0 \
    --batch_size 1024 \
    --local_steps 1 \
    --by_epoch \
    --device cpu \
    --data_dir ../data/adult \
    --chkpts_dir ../chkpts/adult \
    --logs_dir ../logs/adult/training \
    --metadata_dir ../metadata/adult \
    --log_freq 10 \
    --save_freq 1 \
    --num_rounds 100 \
    --seed 42 
```

If you want to save the local models associated to each client, run 

```bash
cd examples/

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
    --data_dir ../data/adult \
    --local_models_dir ../local_models/adult \
    --chkpts_dir ../chkpts/adult \
    --logs_dir ../logs/adult/training \
    --metadata_dir ../metadata/adult \
    --log_freq 10 \
    --save_freq 1 \
    --num_rounds 100 \
    --seed 42 
```



## Contributing
We welcome contributions! To contribute to FedKit-Learn, please follow the guidelines 
outlined in `CONTRIBUTING.md`.
We appreciate bug reports, feature requests, and pull requests.

## License
FedKit-Learn is released under the [Apache License 2.0](LICENSE). Feel free to use, modify, 
and distribute this package in accordance with the terms of the Apache License 2.0.