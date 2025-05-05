# Attribute Inference Attacks for Federated Regression Tasks

This repository is the official code implementation of [Attribute Inference Attacks for Federated Regression Tasks](https://arxiv.org/abs/2411.12697)

Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks. 

## Installation
To clone the repository, run 

```bash
git clone https://github.com/francescodiana99/fedkit-learn.git
cd fedkit-learn
```
Install the package and its dependencies using `setup.py`
```bash
pip install .
```
## Usage
The package includes a set of tools and utilities to simulate federated learning scenarios. More specifically, it provides:
- **Federated Learning Simulator**: Simulate federated learning scenarios with customizable parameters, algorithms, and communication strategies.
- **Federated Attack Simulator** Test the robustness of your federated learning system against various attacks and adversarial scenarios.
- **Federated Datasets**: Access federated datasets or create your own for training and evaluation.

We We provide code to simulate federated training of machine learning models (`federated/`), as well as code to simulate 
machine learning attacks (`attacks/`). Moreover, we implement standard federated learning datasets in `datasets/`. 

## Paper Experiments
To run the experiments in the paper, check `bash_scripts/README.md`. Experiments on a Toy Dataset can be found in `toy_experiments.ipynb`.



## Citation
```
@article{Diana_Marfoq_Xu_Neglia_Giroire_Thomas_2025, title={Attribute Inference Attacks for Federated Regression Tasks}, volume={39}, url={https://ojs.aaai.org/index.php/AAAI/article/view/33787}, DOI={10.1609/aaai.v39i15.33787}, number={15}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Diana, Francesco and Marfoq, Othmane and Xu, Chuan and Neglia, Giovanni and Giroire, Frédéric and Thomas, Eoin}, year={2025}, month={Apr.}, pages={16271-16279} }
```

## Contribution
We welcome contributions! To contribute to FedKit-Learn, please follow the guidelines outlined in CONTRIBUTING.md.
We appreciate bug reports, feature requests, and pull requests.

## License
FedKit-Learn is released under the Apache License 2.0. Feel free to use, modify, and distribute this package in accordance with the terms of the Apache License 2.0.

## Contact

If you have any questions, do not hesitate to write an email at francesco.diana@inria.fr .