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
    - `load_clients_from_chkpt(federated_dataset, args)`: Load clients from given checkpoints.
    - `compute_local_models(federated_dataset, args)`: Compute and save local models.
    - `save_last_round_metadata(all_messages_metadata, metadata_dir)`: Save simulation last round metadata.
    - `save_setup(args, setup_path)`: Save simulation configuration.

    - `main()`: Execute the federated learning simulation.

Usage:
    python run_simulation.py --task_name adult --test_frac 0.2 --scaler standard
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
    --download: Flag for forcing data downloading.
    --sensitive_attribute_id: Sensitive attribute id.
    --cast_float: Flag for casting the target variable to float.
    --target_item: Target item for binary classification with Purchase.
    --n_features: Number of features for binary classification with Purchase.
    --features_correlation_path: Path to the features correlation matrix.
    --state: USA state to extract in Income dataset.
    --mixing_coefficient: Mixing coefficient for the mixing sample distribution in Adult dataset.
    --scale_target: Flag for scaling the target variable in the medical cost dataset.
    --binarize_target: Flag for binarizing the target variable in Income dataset.
    --use_dp: Flag for using differential privacy.
    --noise_multiplier: Noise multiplier for differential privacy.
    --clip_norm: Clipping norm for differential privacy.
    --dp_delta: Delta for differential privacy.
    --dp_epsilon: Epsilon for differential privacy.
    --max_physical_batch_size: Maximum physical batch size for differential privacy.
    --num_active_rounds: Number of expected active rounds for the active attacker.


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
from fedklearn.federated.simulator import FederatedAveraging

from fedklearn.metrics import *

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
                 'income', 'binary_income' ],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification', 'purchase', 'medical_cost',"
             " 'income', 'binary_income', 'linear_income', 'linear_medical_cost'.",
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
        help="Flag for binarizing the marital status in Adult dataset",
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
        "--cast_float",
        action="store_true",
        help="Flag for binary classification",
        default=True
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
        default=None,
        help="Path to the features correlation matrix for binary classification with Purchase"
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
        help="Mixing coefficient for the mixing sample distribution in Income and Adult datasets"
    )

    parser.add_argument(
        "--keep_proportions",
        action="store_true",
        help="Flag for keeping the original proportions of the states  when sampling from Income dataset"
    )

    parser.add_argument(
        "--scale_target",
        action="store_true",
        help="Flag for scaling the target variable"
    )

    parser.add_argument(
        "--binarize_target",
        action="store_true",
        default=False,
            help="Flag for binarizing the target variable in Income dataset"
        )

    # differential privacy args
    parser.add_argument(
        "--use_dp",
        action="store_true",
        default=False,
        help="Flag for training with differential privacy"
    )

    parser.add_argument(
        "--num_active_rounds",
        type=int,
        default=None,
        help="Number of expected active rounds for the active attacker."
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
    if args.task_name in ['toy_classification', 'toy_classification', 'purchase', 'income', 'binary_income', 
                          'linar_income', 'medical_cost', 'linear_medical_cost'] and args.n_tasks is None:
        raise ValueError(
            f"The number of tasks should be specified for {args.task_name} dataset."
        )
    
    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)
        if model_config["model_class"] ==  "LinearLayer":
            use_linear = True
        else:
            use_linear =  False

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
            scale_target=args.scale_target,
            use_linear=use_linear
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
            keep_proportions=args.keep_proportions,
            scale_target=args.scale_target,
            use_linear=use_linear,
            binarize=args.binarize_target,
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
        

    criterion, metric, cast_float = get_trainers_config(args.task_name)
    criterion = criterion.to(args.device)

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
            "init_fn": optim.Adam,
            "betas": (0.99, 0.999)
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
        if args.num_active_rounds is None:
            raise ValueError("The number of expected active rounds should be specified to correctly compute the privacy budget.")
        total_n_epochs = (args.num_rounds + args.num_active_rounds) * args.local_steps
        return DPTrainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=args.device,
            optimizer=optimizer,
            cast_float=cast_float,
            max_physical_batch_size=args.max_physical_batch_size if args.max_physical_batch_size is not None else args.batch_size,
            noise_multiplier=args.noise_multiplier,
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            clip_norm=args.clip_norm,
            epochs=total_n_epochs,
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
            cast_float=cast_float,
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
    """
    Save simulation last round metadata.

    Args:
        all_messages_metadata (dict): Dictionary of all messages metadata.
        metadata_dir (str): Metadata directory.
    """
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

def save_setup(args, data_path, hparams_path):
    """
    Save hyperparameters configuration.
    """
    hp_dict = {
        "task_name": args.task_name,
        "scale_target": args.scale_target,
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "local_steps": args.local_steps,
        "by_epoch": args.by_epoch,
        "num_rounds": args.num_rounds,
        "dp_epsilon": args.dp_epsilon,
        "dp_delta": args.dp_delta,
        "noise_multiplier": args.noise_multiplier,
        "clip_norm": args.clip_norm,
        "max_physical_batch_size": args.max_physical_batch_size,
        "num_active_rounds": args.num_active_rounds,
        "data_path": data_path,
        "chkpts_dir": args.chkpts_dir,
        "model_config_path": args.model_config_path,
        "binarize": args.binarize_target
        }

    with open(hparams_path, "w") as f:
        json.dump(hp_dict, f)

    
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
            raise ValueError("'noise_multiplier', 'dp_epsilon', 'clip_norm', and 'dp_delta' parameters must be specified \
                             when training with differential privacy")

    federated_dataset = initialize_dataset(args, rng)

    logging.info("=" * 100)
    logging.info("Initialize clients..")
    clients = initialize_clients(federated_dataset, args)
    os.makedirs(args.metadata_dir, exist_ok=True)

    if args.compute_local_models:
        logging.info("=" * 100)
        logging.info("Compute local models..")
        os.makedirs(args.local_models_dir,  exist_ok=True)
        local_models_dict, models_trajectory_dict = compute_local_models(federated_dataset, args)

        logging.info("Saving local models metadata..")

        local_models_metadata_path = os.path.join(args.metadata_dir, "last_local.json")
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
    for round_id in tqdm(range(args.num_rounds)):
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

    setup_path = os.path.join(args.metadata_dir, "setup.json")
    save_setup(args, federated_dataset.metadata_path, setup_path)
    logging.info(f"The simulation setup has been saved in {setup_path}")


if __name__ == "__main__":
    main()
