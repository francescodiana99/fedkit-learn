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
    - `initialize_trainer(args)`: Initialize the trainer based on the specified task.
    - `initialize_clients(federated_dataset, args)`: Initialize clients based on the federated
                                                     dataset and command-line arguments.
    - `initialize_simulator(clients, args, rng)`: Initialize the federated averaging simulator.
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
"""

import os
import json
import argparse

import numpy as np

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.models.linear import LinearLayer
from fedklearn.trainer.trainer import Trainer
from fedklearn.federated.client import Client
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
        help="Task name. Possible are 'adult', 'faces', and 'cifar'",
        required=True
    )
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
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Directory to cache data"
    )

    parser.add_argument(
        "--compute_local_models",
        action="store_true",
        help="Flag for training local models"
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
        default=0.9,
        help="Momentum"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024, help="Batch size"
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
        type=int, default=10,
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
    if args.task_name == "adult":
        return FederatedAdultDataset(
            cache_dir=args.data_dir,
            test_frac=args.test_frac,
            drop_nationality=not args.use_nationality,
            scaler_name=args.scaler_name,
            rng=rng
        )
    else:
        raise NotImplementedError(
            f"Dataset initialization for task '{args.task_name}' is not implemented."
        )


def initialize_trainer(args):
    """
    Initialize the trainer based on the specified task.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trainer: Initialized trainer.
    """
    if args.task_name == "adult":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        metric = binary_accuracy_with_sigmoid
        # TODO: infer the input_dimension
        model = LinearLayer(input_dimension=41, output_dimension=1).to(args.device)
        is_binary_classification = True
    else:
        raise NotImplementedError(
            f"Trainer initialization for task '{args.task_name}' is not implemented."
        )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{args.optimizer}' is not implemented."
        )

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
        trainer = initialize_trainer(args)

        train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        logger = SummaryWriter(os.path.join(args.logs_dir, f"client_{task_id}"))

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
         Dict:
            dictionary mapping each client name to the path of its corresponding local model
     """
    models_dict = dict()
    for task_id in tqdm(federated_dataset.task_id_to_name):
        trainer = initialize_trainer(args)

        train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        trainer.fit_epochs(loader=train_loader, n_epochs=args.num_rounds)

        train_loss, train_metric = trainer.evaluate_loader(train_loader)
        test_loss, test_metric = trainer.evaluate_loader(test_loader)

        logging.info("+" * 50)
        logging.info(f"Task ID: {task_id}")
        logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
        logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")
        logging.info("+" * 50)

        path = os.path.join(args.local_models_dir, f"{task_id}.pt")
        path = os.path.abspath(path)
        trainer.save_checkpoint(path)

        models_dict[f"{task_id}"] = path

    return models_dict


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
    global_trainer = initialize_trainer(args)
    global_logger = SummaryWriter(os.path.join(args.logs_dir, "global"))

    simulator = FederatedAveraging(
        clients=clients,
        global_trainer=global_trainer,
        logger=global_logger,
        chkpts_dir=args.chkpts_dir,
        rng=rng,
    )

    return simulator


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

    rng = np.random.default_rng(seed=args.seed)

    federated_dataset = initialize_dataset(args, rng)

    logging.info("=" * 100)
    logging.info("Initialize clients..")
    clients = initialize_clients(federated_dataset, args)

    os.makedirs(args.metadata_dir, exist_ok=True)

    if args.compute_local_models:
        logging.info("=" * 100)
        logging.info("Compute local models..")
        os.makedirs(args.local_models_dir,  exist_ok=True)
        local_models_dict = compute_local_models(federated_dataset, args)

        logging.info("Saving local models metadata..")

        local_models_metadata_path = os.path.join(args.metadata_dir, "local.json")
        with open(local_models_metadata_path, "w") as f:
            json.dump(local_models_dict, f)

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


if __name__ == "__main__":
    main()
