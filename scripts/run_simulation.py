"""Federated Learning Simulation Script

This script performs a federated learning simulation using the specified parameters.
It initializes clients, a global trainer, and a simulator to run the federated learning
process for a given number of rounds.

Command-line arguments can be used to customize the simulation parameters.

Methods/Functions:
    - `none_or_float(value)`: Helper function to convert 'none' to None or float values.
    - `configure_logging(args)`: Set up logging based on verbosity level.
    - `binary_accuracy(y_pred, y)`: Calculate binary accuracy given predictions and ground truth.
    - `parse_args(args_list=None)`: Parse command-line arguments.
    - `initialize_dataset(args, rng)`: Initialize the federated dataset based on the specified task.
    - `initialize_trainer(args, use_dp)`: Initialize the trainer based on the specified task.
    - `initialize_clients(federated_dataset, args)`: Initialize clients based on the federated
                                                     dataset and command-line arguments.
    - `initialize_simulator(clients, args, rng)`: Initialize the federated averaging simulator.
    - `save_last_round_metadata(all_messages_metadata, metadata_dir)`: Save metadata for the last active round.
    - `load_clients_from_chkpt(federated_dataset, args)`: Load clients from given checkpoints.
    - `compute_local_models(federated_dataset, args)`: Compute and save local models.
    - `initialize_active_simulator(clients, args, rng)`: Initialize the active federated averaging simulator.
    - `main()`: Execute the federated learning simulation.

Usage:
    python run_simulation.py --task_name adult --test_frac 0.2 --use_nationality --scaler standard
                             -- optimizer sgd --learning_rate 0.01 --momentum 0.9 --weight_decay 5e-4
                             --batch_size 1024 --local_steps 1 --by_epoch
                             --device cuda --metadata_dir metadata
                             --data_dir data --chkpts_dir chkpts --logs_dir logs
                              --log_freq 10 --save_freq 1
                             --num_rounds 100 --seed 42

Options:
    --task_name: Task name. Possible values are 'adult', 'faces', and 'cifar'.
    --test_frac: Fraction of the test samples; it should be a float between 0 and 1.
                 Treated as None if not specified.
    --use_nationality: If chosen, the nationality column will be kept; otherwise, it is dropped.
    --scaler: Name of the scaler used to scale numerical features.
          Default is 'standard'. It can be 'min_max' or 'standard'.
    --learning_rate: Learning rate for the optimizer.
    --momentum: Momentum for the optimizer.
    --weight_decay: Weight decay for the optimizer.
    --batch_size: Batch size for training and testing.
    --local_steps: Number of local steps performed by each client during simulation.
    --by_epoch: Flag for training by epoch instead of steps.
    --device: Device for computation, either 'cpu' or 'cuda'.
    --data_dir: Directory to cache data.
    --chkpts_dir: Checkpoints directory for saving model checkpoints.
    --logs_dir: Logs directory for storing TensorBoard logs.
    --metadata_dir: Metadata dir for saving local models paths and messages metadata.
    --log_freq: Logging frequency, specifying how often to log simulation progress.
    --save_freq: Saving frequency, specifying how often to save model checkpoints.
    --num_rounds: Number of simulation rounds.
    --seed: Random seed for reproducibility.
    --binarize_marital_status: Flag for binarizing the marital status.
    --force_generation: Flag for forcing the generation of the dataset.
    --sensitive_attribute_id: Sensitive attribute id.
    --is_binary_classification: Flag for binary classification.
    --target_item: Target item for binary classification with Purchase.
    --n_features: Number of features for binary classification with Purchase.
    --features_correlation_path: Path to the features correlation matrix.
    --state: USA state to extract in Income dataset.
    --mixing_coefficient: Mixing coefficient for the mixing sample distribution in Adult dataset.
    --scale_target: Flag for scaling the target variable in the medical cost dataset.
    --active_server: Flag for using an active server that computes updates.
    --alpha: Alpha parameter for the Adam optimizer in the active attack scenario.
    --beta1: Beta1 parameter for the Adam optimizer in the active attack scenario.
    --beta2: Beta2 parameter for the Adam optimizer in the active attack scenario.
    --epsilon: Epsilon parameter for the Adam optimizer in the active attack scenario.
    --attacked_round: Round in which the active attack is performed.
    --use_dp: Flag for using differential privacy.
    --noise_multiplier: Noise multiplier for differential privacy.
    --clip_norm: Clipping norm for differential privacy.
    --dp_delta: Delta for differential privacy.
    --dp_epsilon: Epsilon for differential privacy.
    --max_physical_batch_size: Maximum physical batch size for differential privacy.

"""
import argparse
import logging
import os
import pathlib
from datetime import datetime

import numpy as np

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedklearn.datasets.income.income import FederatedIncomeDataset
from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.datasets.medical_cost.medical_cost import FederatedMedicalCostDataset
from fedklearn.datasets.purchase.purchase import FederatedPurchaseDataset, FederatedPurchaseBinaryClassificationDataset
from fedklearn.datasets.toy.toy import FederatedToyDataset
from fedklearn.models.linear import LinearLayer
from fedklearn.trainer.trainer import Trainer, DPTrainer
from fedklearn.federated.client import Client, DPClient
from fedklearn.federated.simulator import FederatedAveraging, ActiveAdamFederatedAveraging

from fedklearn.metrics import *
import optuna

from utils import *


def parse_args(args_list=None):
    """
    Parse command-line arguments.

     Args:
        args_list (list, optional): List of command-line arguments. If None, sys.argv is used.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task_name",
        type=str,
        choices=['adult', 'toy_regression', 'toy_classification', 'purchase', 'purchase_binary', 'medical_cost',
                 'income', 'binary_income', 'linear_income', 'linear_medical_cost', 'dp_income', 'dp_medical_cost'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification', 'purchase', 'medical_cost',"
             " 'income', 'binary_income', 'linear_income', 'linear_medical_cost', 'dp_income', 'dp_medical_cost'.",
        required=True
    )

    # Args for adult dataset
    parser.add_argument(
        '--test_frac',
        help='Fraction of the test samples; it should be a float between 0 and 1.'
             'Treated as None if not specified',
        type=none_or_float,
        default=None
    )
    parser.add_argument(
        '--use_nationality',
        help='If chosen the nationality column will be kept; otherwise, it is dropped',
        action='store_true'
    )
    parser.add_argument(
        "--scaler_name",
        type=str,
        default="standard",
        help="Name of the scaler used to scale numerical features."
             "Default is 'standard'. It can be 'min_max' or 'standard'."
    )

    # Args for toy dataset
    parser.add_argument(
        "--use_bias",
        action="store_true",
        help="If selected, a bias term to the linear model behind the toy dataset."
    )

    parser.add_argument(
        "--split_clients",
        action="store_true",
        help="If selected, split the data of Toy dataset on two groups"
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=None,
        help="Number of tasks"
    )

    parser.add_argument(
        "--n_task_samples",
        type=int,
        default=None,
        help="Number of samples per task"
    )

    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=30,
        help="Number of training samples"
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=1_000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--n_numerical_features",
        type=int,
        default=1,
        help="Number of numerical features"
    )
    parser.add_argument(
        "--n_binary_features",
        type=int,
        default=1,
        help="Number of binary features"
    )
    parser.add_argument(
        "--sensitive_attribute_type",
        type=str,
        choices=["binary", "numerical"],
        default="binary",
        help="Type of sensitive attribute"
    )

    parser.add_argument(
        "--split_criterion",
        type=str,
        help="Criterion for splitting the dataset",
    )
    parser.add_argument(
        "--sensitive_attribute_weight",
        type=float,
        default=0.5,
        help="Weight of sensitive attribute"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Level of noise"
    )

    # Federated learning args
    parser.add_argument(
        "--compute_local_models",
        action="store_true",
        help="Flag for training local models"
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to the model configuration file",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.,
        help="Momentum"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size"
    )

    parser.add_argument(
        "--local_steps",
        type=int,
        default=1,
        help="Local steps"
    )
    parser.add_argument(
        "--by_epoch",
        action="store_true",
        help="Flag for training by epoch"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=100,
        help="Number of simulation rounds"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )

    # Directories and logging args
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Directory to cache data"
    )
    parser.add_argument(
        "--local_models_dir",
        type=str,
        default="./local_models",
        help="Directory to save local models dir"
    )
    parser.add_argument(
        "--chkpts_dir",
        type=str,
        default="./chkpts",
        help="Checkpoints directory"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Logs directory"
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="metadata",
        help="Metadata directory"
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=10,
        help="Logging frequency"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Saving frequency"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Force data downloading'
    )

    parser.add_argument(
        '-v', '--verbose',
        help='Increase verbosity level. Repeat for more detailed log messages.',
        action='count',
        default=0
    )

    parser.add_argument(
        '-q', '--quiet',
        help='Decrease verbosity level. Repeat for less detailed log messages.',
        action='count',
        default=0
    )

    parser.add_argument(
        "--binarize_marital_status",
        action="store_true",
        help="Flag for binarizing the marital status",
        default=False
    )

    parser.add_argument(
        "--force_generation",
        action="store_true",
        help="Flag for forcing the generation of the dataset",
        default=False
    )

    parser.add_argument(
        "--sensitive_attribute_id",
        type=str,
        default=0,
        help="Sensitive attribute id"
    )

    parser.add_argument(
        "--is_binary_classification",
        action="store_true",
        help="Flag for binary classification",
        default=False
    )

    parser.add_argument(
        "--target_item",
        default=None,
        help="Target item for binary classification with Purchase"
    )

    parser.add_argument(
        "--n_features",
        type=int,
        default=20,
        help="Number of features for binary classification with Purchase"
    )

    parser.add_argument(
        "--features_correlation_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="USA state to extract in Income dataset"
    )

    parser.add_argument(
        "--mixing_coefficient",
        type=float,
        default=0,
        help="Mixing coefficient for the mixing sample distribution in Adult dataset"
    )

    parser.add_argument(
        "--keep_proportions",
        action="store_true",
        help="Flag for keeping the proportions of the states in the Income dataset"
    )

    parser.add_argument(
        "--scale_target",
        action="store_true",
        help="Flag for scaling the target variable in the medical cost dataset"
    )
    # active server args
    parser.add_argument(
        "--num_active_rounds",
        type=int,
        default=0,
        help="Number of active rounds"
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials for the hyperparameter optimization"
    )

    parser.add_argument(
        "--hparams_config_path",
        type=str,
        default="../configs/hyperparameters.json",
        help="Path to the hyperparameters configuration file"
    )

    parser.add_argument(
        "--attacked_task",
        type=str,
        default=None,
        help="If set, the active attack is performed on the specified task"
    )

    parser.add_argument(
        "--use_norm",
        action="store_true",
        help="Flag for using the norm as proxy for tuning the active attack"
    )

    # differential privacy args
    parser.add_argument(
        "--use_dp",
        action="store_true",
        default=False,
        help="Flag for using differential privacy"
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Noise multiplier for differential privacy"
    )

    parser.add_argument(
        "--clip_norm",
        type=float,
        default=None,
        help="Clipping norm for differential privacy"
    )

    parser.add_argument(
        "--dp_delta",
        type=float,
        default=None,
        help="Delta for differential privacy"
    )

    parser.add_argument(
        "--dp_epsilon",
        type=float,
        default=None,
        help="Epsilon for differential privacy"
    )

    parser.add_argument(
        "--max_physical_batch_size",
        type=int,
        default=None,
        help="Maximum physical batch size for differential privacy"
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def initialize_dataset(args, rng):
    """
    Initialize the federated dataset based on the specified task.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        FederatedDataset: Initialized federated dataset.
    """
    if args.task_name in ['toy_classification', 'toy_classification', 'purchase', 'income'] and args.n_tasks is None:
        raise ValueError(
            f"The number of tasks should be specified for {args.task_name} dataset."
        )

    if args.task_name == "adult":
        return FederatedAdultDataset(
            cache_dir=args.data_dir,
            test_frac=args.test_frac,
            drop_nationality=not args.use_nationality,
            scaler_name=args.scaler_name,
            rng=rng,
            split_criterion=args.split_criterion,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            download=args.download,
            force_generation=args.force_generation,
            seed=args.seed,
            binarize_marital_status=args.binarize_marital_status,
            sensitive_attribute_id=args.sensitive_attribute_id,
            device=args.device,
            mixing_coefficient=args.mixing_coefficient
        )
    elif args.task_name == "toy_regression":
        return FederatedToyDataset(
            n_tasks=args.n_tasks,
            n_train_samples=args.n_train_samples,
            n_test_samples=args.n_test_samples,
            problem_type="regression",
            bias=args.use_bias,
            n_numerical_features=args.n_numerical_features,
            n_binary_features=args.n_binary_features,
            sensitive_attribute_type=args.sensitive_attribute_type,
            sensitive_attribute_weight=args.sensitive_attribute_weight,
            noise_level=args.noise_level,
            force_generation=args.force_generation,
            allow_generation=True,
            cache_dir=args.data_dir,
            rng=rng,
            split_clients=args.split_clients
        )
    elif args.task_name == "toy_classification":
        return FederatedToyDataset(
            n_tasks=args.n_tasks,
            n_train_samples=args.n_train_samples,
            n_test_samples=args.n_test_samples,
            problem_type="classification",
            bias=args.use_bias,
            n_numerical_features=args.n_numerical_features,
            n_binary_features=args.n_binary_features,
            sensitive_attribute_type=args.sensitive_attribute_type,
            sensitive_attribute_weight=args.sensitive_attribute_weight,
            noise_level=args.noise_level,
            force_generation=args.force_generation,
            allow_generation=True,
            cache_dir=args.data_dir,
            rng=rng,
            split_clients=args.split_clients
        )

    elif args.task_name == "purchase":
        return FederatedPurchaseDataset(
            cache_dir=args.data_dir,
            download=args.download,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            rng=rng,
            split_criterion=args.split_criterion,
            test_frac=args.test_frac
        )

    elif args.task_name == "purchase_binary":
        return FederatedPurchaseBinaryClassificationDataset(
            cache_dir=args.data_dir,
            download=args.download,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            rng=rng,
            split_criterion=args.split_criterion,
            test_frac=args.test_frac,
            target_item=args.target_item,
            n_features=args.n_features,
            sensitive_attribute=args.sensitive_attribute_id,
            feature_correlation_path=args.features_correlation_path
        )
    elif args.task_name =="medical_cost":
        return FederatedMedicalCostDataset(
            cache_dir=args.data_dir,
            download=args.download,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            rng=rng,
            split_criterion=args.split_criterion,
            test_frac=args.test_frac,
            scaler=args.scaler_name,
            scale_target=args.scale_target
        )

    elif args.task_name =="linear_medical_cost":
        return FederatedMedicalCostDataset(
            cache_dir=args.data_dir,
            download=args.download,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            rng=rng,
            split_criterion=args.split_criterion,
            test_frac=args.test_frac,
            scaler=args.scaler_name,
            scale_target=args.scale_target,
            use_linear=True
        )

    elif args.task_name =="dp_medical_cost":
        return FederatedMedicalCostDataset(
            cache_dir=args.data_dir,
            download=args.download,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            rng=rng,
            split_criterion=args.split_criterion,
            test_frac=args.test_frac,
            scaler=args.scaler_name,
            scale_target=args.scale_target,
            use_dp=True
        )


    if args.task_name == "income":
        return FederatedIncomeDataset(
            cache_dir=args.data_dir,
            download=args.download,
            test_frac=args.test_frac,
            scaler_name=args.scaler_name,
            drop_nationality=not args.use_nationality,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            seed=args.seed,
            rng=rng,
            split_criterion=args.split_criterion,
            state=args.state,
            mixing_coefficient=args.mixing_coefficient,
            keep_proportions=args.keep_proportions
        )

    if args.task_name == "binary_income":
        return FederatedIncomeDataset(
            cache_dir=args.data_dir,
            download=args.download,
            test_frac=args.test_frac,
            scaler_name=args.scaler_name,
            drop_nationality=not args.use_nationality,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            seed=args.seed,
            rng=rng,
            split_criterion=args.split_criterion,
            state=args.state,
            mixing_coefficient=args.mixing_coefficient,
            keep_proportions=args.keep_proportions,
            binarize=True
        )

    if args.task_name == "linear_income":
        return FederatedIncomeDataset(
            cache_dir=args.data_dir,
            download=args.download,
            test_frac=args.test_frac,
            scaler_name=args.scaler_name,
            drop_nationality=not args.use_nationality,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            seed=args.seed,
            rng=rng,
            split_criterion=args.split_criterion,
            state=args.state,
            mixing_coefficient=args.mixing_coefficient,
            keep_proportions=args.keep_proportions,
            use_linear=True
        )

    if args.task_name == "dp_income":
        return FederatedIncomeDataset(
            cache_dir=args.data_dir,
            download=args.download,
            test_frac=args.test_frac,
            scaler_name=args.scaler_name,
            drop_nationality=not args.use_nationality,
            force_generation=args.force_generation,
            n_tasks=args.n_tasks,
            n_task_samples=args.n_task_samples,
            seed=args.seed,
            rng=rng,
            split_criterion=args.split_criterion,
            state=args.state,
            mixing_coefficient=args.mixing_coefficient,
            keep_proportions=args.keep_proportions,
            use_dp=True
        )

    else:
        raise NotImplementedError(
            f"Dataset initialization for task '{args.task_name}' is not implemented."
        )


def initialize_trainer(args, use_dp=False, train_loader=None):
    """
    Initialize the trainer based on the specified task.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trainer: Initialized trainer.
    """
    model_config_dir = pathlib.Path("../fedklearn/configs")

    if model_config_dir not in pathlib.Path(args.model_config_path).parents:
        raise ValueError(f"Model configuration file should be placed in {model_config_dir}")
    else:
        model = initialize_model(args.model_config_path)

    if args.task_name == "adult":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        metric = binary_accuracy_with_sigmoid
        is_binary_classification = True
    elif args.task_name == "toy_classification":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        metric = binary_accuracy_with_sigmoid
        is_binary_classification = True

    elif args.task_name == "toy_regression":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_squared_error
        is_binary_classification = False

    elif args.task_name == "purchase":
        criterion = nn.CrossEntropyLoss().to(args.device)
        metric = multiclass_accuracy
        is_binary_classification = False

    elif args.task_name == "purchase_binary":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        metric = binary_accuracy_with_sigmoid
        is_binary_classification = True

    elif args.task_name == "medical_cost":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

    elif args.task_name == "linear_medical_cost":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

    elif args.task_name == "dp_medical_cost":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

    elif args.task_name == "income":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

    elif args.task_name == "binary_income":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        metric = binary_accuracy_with_sigmoid
        is_binary_classification = True

    elif args.task_name == "linear_income":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

    elif args.task_name == "dp_income":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

    else:
        raise NotImplementedError(
            f"Trainer initialization for task '{args.task_name}' is not implemented."
        )

    if args.optimizer == "sgd":
        optimizer_params = {
            "lr": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "init_fn": optim.SGD
        }

        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer_params = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "init_fn": optim.Adam
        }

        optimizer = optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{args.optimizer}' is not implemented."
        )

    if use_dp:

        return DPTrainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=args.device,
            optimizer=optimizer,
            is_binary_classification=is_binary_classification,
            max_physical_batch_size=args.max_physical_batch_size if args.max_physical_batch_size is not None else args.batch_size,
            noise_multiplier=args.noise_multiplier,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            clip_norm=args.clip_norm,
            epochs=args.num_rounds * args.local_steps,
            train_loader=train_loader,
            optimizer_init_dict=optimizer_params,
            rng=torch.Generator(device=args.device).manual_seed(args.seed)
    )
    else:
        return Trainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=args.device,
            optimizer=optimizer,
            is_binary_classification=is_binary_classification,
        )


def initialize_clients(federated_dataset, args):
    """
     Initialize clients based on the federated dataset and command-line arguments.

     Args:
         federated_dataset (FederatedDataset): Initialized federated dataset.
         args (argparse.Namespace): Parsed command-line arguments.

     Returns:
         list of Client: List of initialized clients.
     """
    clients = []
    for task_id in federated_dataset.task_id_to_name:
        train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        trainer = initialize_trainer(args, use_dp=args.use_dp, train_loader=train_loader)

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{task_id}"))

        if args.use_dp:
            client= DPClient(
                trainer=trainer,
                train_loader=train_loader,
                test_loader=test_loader,
                local_steps=args.local_steps,
                by_epoch=args.by_epoch,
                logger=logger,
            )
        else:

            client = Client(
                trainer=trainer,
                train_loader=train_loader,
                test_loader=test_loader,
                local_steps=args.local_steps,
                by_epoch=args.by_epoch,
                logger=logger,
            )

        clients.append(client)

    return clients


def compute_local_models(federated_dataset, args):
    """
     Compute and Save Local Models

     Args:
         federated_dataset (FederatedDataset): Initialized federated dataset.
         args (argparse.Namespace): Parsed command-line arguments.

     Returns:
            Dict: Dictionary of local models paths.
            Dict: Dictionary of local models training trajectories.
     """
    models_trajectory_dict = dict()
    for task_id in federated_dataset.task_id_to_name:
        models_trajectory_dict[f"{task_id}"] = dict()

    models_dict = dict()
    for task_id in tqdm(federated_dataset.task_id_to_name):
        trainer = initialize_trainer(args, use_dp=False)

        train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        for step in range(args.num_rounds):
            trainer.fit_epoch(train_loader)

            if step % args.save_freq == 0:

                os.makedirs(os.path.join(args.local_models_dir, task_id), exist_ok=True)

                path = os.path.join(args.local_models_dir, task_id, f"{step}.pt")
                path = os.path.abspath(path)
                trainer.save_checkpoint(path)
                models_trajectory_dict[f"{task_id}"][f"{step}"] = path

            if trainer.lr_scheduler is not None:
                trainer.lr_scheduler.step()


        train_loss, train_metric = trainer.evaluate_loader(train_loader)
        test_loss, test_metric = trainer.evaluate_loader(test_loader)

        logging.info("+" * 50)
        logging.info(f"Task ID: {task_id}")
        logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
        logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")
        logging.info("+" * 50)

        local_model_path = os.path.join(args.local_models_dir, task_id,  f"{args.num_rounds - 1}.pt")
        local_model_path = os.path.abspath(local_model_path)

        models_dict[f"{task_id}"] = local_model_path

    return models_dict, models_trajectory_dict


def initialize_simulator(clients, args, rng):
    """
    Initialize the federated averaging simulator.

    Args:
        clients (list of Client): List of initialized clients.
        args (argparse.Namespace): Parsed command-line arguments.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        FederatedAveraging: Initialized federated averaging simulator.
    """
    global_trainer = initialize_trainer(args, use_dp=False)
    global_logger = SummaryWriter(os.path.join(args.logs_dir, "global"))

    simulator = FederatedAveraging(
        clients=clients,
        global_trainer=global_trainer,
        logger=global_logger,
        chkpts_dir=args.chkpts_dir,
        use_dp=args.use_dp,
        rng=rng,
    )

    return simulator


def save_last_round_metadata(all_messages_metadata, metadata_dir):
    last_saved_round_id = max(map(int, all_messages_metadata["global"].keys()))
    _random_key = list(all_messages_metadata["global"].keys())[0]
    last_saved_round_id = int(last_saved_round_id) if isinstance(_random_key, int) else str(last_saved_round_id)

    last_clients_messages_metadata = dict()
    last_server_messages_metadata = dict()

    for client_id in all_messages_metadata:
        if str(client_id).isdigit():
            last_clients_messages_metadata[client_id] = all_messages_metadata[client_id][last_saved_round_id]
            last_server_messages_metadata[client_id] = all_messages_metadata["global"][last_saved_round_id]

    last_models_metadata_path = os.path.join(metadata_dir, "last.json")
    with open(last_models_metadata_path, "w") as f:
        json.dump(last_clients_messages_metadata, f)

    logging.info(f"Last models sent by the client saved to {last_models_metadata_path}")

    server_models_metadata_path = os.path.join(metadata_dir, "server.json")
    with open(server_models_metadata_path, "w") as f:
        json.dump(last_server_messages_metadata, f)

    logging.info(f"Last models sent by the server saved to {server_models_metadata_path}")


def load_clients_from_chkpt(federated_dataset, args):
    """
    Load clients from given checkpoints.

    Args:
        federated_dataset (FederatedDataset): Federated dataset for the specific task.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        list: List of clients.
    """
    set_seeds(args.seed)
    clients = []
    with open(os.path.join(args.metadata_dir, "last.json"), "r") as f:
        models_metadata_dict = json.load(f)
    if args.attacked_task is None:
        for task_id in federated_dataset.task_id_to_name:

            train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
            test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            logger = SummaryWriter(os.path.join(args.logs_dir, f"{task_id}"))

            if args.use_dp:
                trainer = initialize_trainer(args, use_dp=args.use_dp, train_loader=train_loader)
                trainer.load_checkpoint(models_metadata_dict[task_id])
                client = DPClient(trainer=trainer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                local_steps=args.local_steps,
                                by_epoch=args.by_epoch,
                                logger=logger,
                                name=task_id
                                )
            else:
                trainer = initialize_trainer(args)
                trainer.load_checkpoint(models_metadata_dict[task_id])

                client = Client(trainer=trainer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                local_steps=args.local_steps,
                                by_epoch=args.by_epoch,
                                logger=logger,
                                name=task_id
                                )
            clients.append(client)
    else:

        train_dataset = federated_dataset.get_task_dataset(args.attacked_task, mode="train")
        test_dataset = federated_dataset.get_task_dataset(args.attacked_task, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{args.attacked_task}"))

        if args.use_dp:
            trainer = initialize_trainer(args, use_dp=args.use_dp, train_loader=train_loader)
            trainer.load_checkpoint(models_metadata_dict[args.attacked_task])
            client = DPClient(trainer=trainer,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            local_steps=args.local_steps,
                            by_epoch=args.by_epoch,
                            logger=logger,
                            name=args.attacked_task
                              )
        else:
            trainer = initialize_trainer(args)
            trainer.load_checkpoint(models_metadata_dict[args.attacked_task])


            client = Client(trainer=trainer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        local_steps=args.local_steps,
                        by_epoch=args.by_epoch,
                        logger=logger,
                        name=args.attacked_task
                      )

        clients.append(client)
    return clients


def save_last_round_active_metadata(args, all_messages_metadata, metadata_dir):
    if args.attacked_task is None:
        last_saved_round_id = max(map(int, all_messages_metadata["0"].keys()))
        _random_key = list(all_messages_metadata["0"].keys())[0]
    else:
        last_saved_round_id = max(map(int, all_messages_metadata[args.attacked_task].keys()))
        _random_key = list(all_messages_metadata[args.attacked_task].keys())[0]
    last_saved_round_id = int(last_saved_round_id) if isinstance(_random_key, int) else str(last_saved_round_id)
    last_clients_messages_metadata = dict()
    last_server_messages_metadata = dict()
    for client_id in all_messages_metadata:
        if str(client_id).isdigit():
            last_clients_messages_metadata[client_id] = all_messages_metadata[client_id][last_saved_round_id]
            last_server_messages_metadata[client_id] = all_messages_metadata["server"][client_id][last_saved_round_id]

    attacked_round = args.num_rounds - args.num_active_rounds - 1
    if args.use_norm:
        last_models_metadata_path = os.path.join(metadata_dir, f"last_active_{attacked_round}_norm.json")
    else:
        last_models_metadata_path = os.path.join(metadata_dir, f"last_active_{attacked_round}.json")
    last_models_dict = read_dict(last_models_metadata_path)
    last_models_dict[f'{attacked_round}'] = last_clients_messages_metadata
    with open(last_models_metadata_path, "w") as f:
        json.dump(last_models_dict, f)

    logging.info(f"Last models sent by the client saved to {last_models_metadata_path}")


def initialize_active_simulator(clients, rng, beta1, beta2, alpha, args):
    """
    Initialize the active federated averaging simulator.

    Args:
        clients (list of Client): List of initialized clients.
        rng (numpy.random.Generator): Random number generator.
        beta1 (float): Beta1 parameter for Adam.
        beta2 (float): Beta2 parameter for Adam.
        alpha (float): Alpha parameter for the active attack.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        ActiveAdamFederatedAveraging: Initialized active federated averaging simulator.
    """
    global_trainer = initialize_trainer(args, use_dp=False)
    with open(os.path.join(args.metadata_dir, "server.json")) as f:
        messages_metadata_dict= json.load(f)
    global_chkpt = messages_metadata_dict["0"]
    global_trainer.load_checkpoint(global_chkpt)
    global_logger = SummaryWriter(os.path.join(args.logs_dir, "active", "global"))

    attacked_round = args.num_rounds - args.num_active_rounds - 1

    simulator = ActiveAdamFederatedAveraging(
        clients=clients,
        global_trainer=global_trainer,
        logger=global_logger,
        chkpts_dir=args.chkpts_dir,
        rng=rng,
        beta1=beta1,
        beta2=beta2,
        epsilon=1e-8,
        alpha=alpha,
        attacked_round=attacked_round,
        use_dp=args.use_dp,
    )
    return simulator


def objective(trial, federated_dataset, rng, args):
    """
    Initialize the objective function for the hyperparameter optimization using Optuna.
    For additional details, please refer to the Optuna documentation:
    https://optuna.readthedocs.io/en/stable/index.html

    Args:
        trial (optuna.Trial): Optuna trial object.
        federated_dataset (FederatedDataset): Initialized federated dataset.
        rng (numpy.random.Generator): Random number generator.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        float: Objective value.
        """
    with open(args.hparams_config_path, "r") as f:
        hparams_dict = json.load(f)
    beta1 = trial.suggest_float("beta1", hparams_dict['beta1'][0], hparams_dict['beta1'][1])
    beta2 = trial.suggest_float("beta2", hparams_dict['beta2'][0], hparams_dict['beta2'][1])
    alpha = trial.suggest_float("alpha", hparams_dict['alpha'][0], hparams_dict['alpha'][1], log=True)

    clients = load_clients_from_chkpt(federated_dataset, args)
    simulator = initialize_active_simulator(clients=clients, rng=rng, beta1=beta1, beta2=beta2, alpha=alpha, args=args)

    logging.info(f"Running active simulation with beta1={beta1}, beta2={beta2}, alpha={alpha}..")
    logging.info('Initial logs..')
    simulator.write_logs(display_only=True)

    for round_id in tqdm(range(args.num_active_rounds)):
        logs_flag = (round_id % args.log_freq == 0)
        simulator.simulate_active_round(save_chkpts=False, save_logs=logs_flag)

    logging.info('Search results..')
    train_loss, _, _, _ = simulator.write_logs(display_only=True)
    pseudo_grad_norm = simulator.compute_pseudo_grad_norm()

    logging.info(f'Pseudo-grad norm: {pseudo_grad_norm}')
    if args.use_norm:
        return pseudo_grad_norm
    else:
        return train_loss


def main():
    """
    Execute the federated learning simulation.

    This function initializes the federated dataset, clients, and simulator based on the
    provided command-line arguments. It then runs the simulation for the specified number
    of rounds, saving model checkpoints and logs at specified intervals. Finally, it saves
    the messages' metadata.

    Returns:
        None
    """
    args = parse_args()

    configure_logging(args)

    set_seeds(seed=args.seed)

    rng = np.random.default_rng(seed=args.seed)

    if args.use_dp:
        if (args.noise_multiplier is None and args.dp_epsilon is None) or args.clip_norm is None or args.dp_delta is None:
            raise ValueError("'noise_multiplier', 'dp_epsilon'., 'clip_norm', and 'dp_delta' parameters must be specified \
                             when training with differential privacy")

    if args.num_rounds <= args.num_active_rounds:
        raise ValueError("The number of simulation rounds should be greater than the number of attack rounds")

    federated_dataset = initialize_dataset(args, rng)

    logging.info("=" * 100)
    logging.info("Initialize clients..")
    clients = initialize_clients(federated_dataset, args)
    os.makedirs(args.metadata_dir, exist_ok=True)

    logging.info("=" * 100)
    logging.info("Saving models configuration metadata..")
    model_config_metadata_path = os.path.join(args.metadata_dir, "model_config.json")
    with open(model_config_metadata_path, "w") as f:
        json.dump({"model_config":args.model_config_path}, f)

    if args.compute_local_models:
        logging.info("=" * 100)
        logging.info("Compute local models..")
        os.makedirs(args.local_models_dir,  exist_ok=True)
        local_models_dict, models_trajectory_dict = compute_local_models(federated_dataset, args)

        logging.info("Saving local models metadata..")

        local_models_metadata_path = os.path.join(args.metadata_dir, "local.json")
        local_models_trajectory_path = os.path.join(args.metadata_dir, "local_trajectories.json")
        with open(local_models_metadata_path, "w") as f:
            json.dump(local_models_dict, f)
        with open(local_models_trajectory_path, "w") as f:
            json.dump(models_trajectory_dict, f)

    logging.info("=" * 100)
    logging.info("Initialize simulator..")
    simulator = initialize_simulator(clients, args, rng)

    logging.info("=" * 100)
    logging.info("Write initial logs..")
    simulator.write_logs()

    logging.info("=" * 100)
    logging.info("Run simulation..")
    for round_id in tqdm(range(args.num_rounds  - args.num_active_rounds)):
        logs_flag = (round_id % args.log_freq == 0)
        chkpts_flag = (round_id % args.save_freq == 0)
        simulator.simulate_round(save_chkpts=chkpts_flag, save_logs=logs_flag)

    logging.info("=" * 100)
    logging.info("Saving simulation results..")

    os.makedirs(os.path.dirname(args.metadata_dir), exist_ok=True)
    messages_metadata_path = os.path.join(args.metadata_dir, "federated.json")
    with open(messages_metadata_path, "w") as f:
        json.dump(simulator.messages_metadata, f)

    logging.info(f"The messages metadata dictionary has been saved in {messages_metadata_path}")
    save_last_round_metadata(simulator.messages_metadata, args.metadata_dir)

    # This part is only for the active simulation
    if args.num_active_rounds > 0:
        logging.info("Simulating active rounds...")

        attacked_round = args.num_rounds - args.num_active_rounds - 1


        if args.hparams_config_path is None:
            raise ValueError("Hyperparameters configuration file is not provided.")

        if not os.path.exists(args.hparams_config_path):
            raise FileNotFoundError(f"Hyperparameters configuration file not found at '{args.hparams_config_path}'.")

        logging.info("=" * 100)
        logging.info("Launch hyperparameter optimization using Optuna..")
        abs_log_dir = os.path.abspath(os.path.join(args.logs_dir, "active"))
        os.makedirs(abs_log_dir, exist_ok=True)
        storage_name = f"sqlite:////{abs_log_dir}/hp_dashboard_{attacked_round}.db"

        study = optuna.create_study(direction="minimize",
                                    storage=storage_name,
                                    load_if_exists=True, study_name=f'{datetime.now()}')
        study.optimize(lambda trial: objective(trial, federated_dataset, rng, args), n_trials=args.n_trials)

        best_params = study.best_params

        logging.info("=" * 100)
        logging.info(f"Best hyperparameters: {study.best_params}")
        logging.info(f"Optimization results saved in: {abs_log_dir}/hp_dashboard_{attacked_round}.db")
        logging.info("=" * 100)

        logging.info("Running active simulation with the best hyperparameters...")

        logging.info("Loading clients from checkpoints...")

        clients = load_clients_from_chkpt(federated_dataset=federated_dataset, args=args)

        logging.info("=" * 100)
        logging.info("Write initial logs..")
        simulator.write_logs()

        logging.info("=" * 100)
        logging.info('Run active simulation...')
        simulator = initialize_active_simulator(clients=clients, rng=rng, beta1=best_params['beta1'],
                                                beta2=best_params['beta2'], alpha=best_params['alpha'], args=args)

        for round_id in tqdm(range(args.num_active_rounds)):
            logs_flag = (round_id % args.log_freq == 0)
            chkpts_flag = (round_id % args.save_freq == 0)

            simulator.simulate_active_round(save_chkpts=chkpts_flag, save_logs=logs_flag)
        logging.info('Final Results..')
        simulator.write_logs(display_only=True)

        logging.info("=" * 100)
        logging.info("Saving simulation results..")
        os.makedirs(os.path.dirname(args.metadata_dir), exist_ok=True)

        if args.use_norm:
            messages_metadata_path = os.path.join(args.metadata_dir, f"active_norm_{attacked_round}.json")
        else:
            messages_metadata_path = os.path.join(args.metadata_dir, f"active_{attacked_round}.json")
        with open(messages_metadata_path, "w") as f:
            json.dump(simulator.messages_metadata, f)

        logging.info(f"The messages metadata dictionary has been saved in {messages_metadata_path}")

        save_last_round_active_metadata(all_messages_metadata=simulator.messages_metadata,
                                        metadata_dir=args.metadata_dir,
                                        args=args)

if __name__ == "__main__":
    main()
