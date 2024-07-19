import argparse
import logging
import os
import pathlib
import shutil
import optuna
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
from fedklearn.trainer.trainer import Trainer
from fedklearn.federated.client import Client
from fedklearn.federated.simulator import FederatedAveraging, ActiveAdamFederatedAveraging

from fedklearn.metrics import *

from utils import *

def parse_args(args_list=None):
    """
    Parse the arguments for the script.
    Args:
        args_list (list, optional): List of command-line arguments. If None, sys.argv is used.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        type=str,
        choices=['adult', 'toy_regression', 'toy_classification', 'purchase', 'purchase_binary', 'medical_cost',
                 'income'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification', 'purchase', 'medical_cost',"
             " 'income'.",
        required=True
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
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size"
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
        '--test_best_hyperparams',
        action='store_true',
        default=False,
        help='If True, test the best hyperparameters found by Optuna.'
        )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def initialize_trainer(args, learning_rate, weight_decay, beta1, beta2):
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

    elif args.task_name == "income":
        criterion = nn.MSELoss().to(args.device)
        metric = mean_absolute_error
        is_binary_classification = False

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
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2)


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


def objective(trial, train_loader, test_loader, task_id, args):
    """
    Initialize the objective function for the hyperparameter optimization using Optuna.
    For additional details, please refer to the Optuna documentation:
    https://optuna.readthedocs.io/en/stable/index.html

    Args:
        trial (optuna.Trial): Optuna trial object.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        float: Objective value.
     """
    with open(args.hparams_config_path, "r") as f:
        hparams_dict = json.load(f)
    beta1 = trial.suggest_float("beta1", hparams_dict['beta1'][0], hparams_dict['beta1'][1])
    beta2 = trial.suggest_float("beta2", hparams_dict['beta2'][0], hparams_dict['beta2'][1])
    lr = trial.suggest_float("lr", hparams_dict['lr'][0], hparams_dict['lr'][1], log=True)
    weight_decay = trial.suggest_float("weight_decay", hparams_dict['weight_decay'][0], hparams_dict['weight_decay'][1],
                                       log=True)

    trainer = initialize_trainer(args, learning_rate=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)

    for round in range(args.num_rounds):
        trainer.fit_epoch(train_loader)
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()

    train_loss, train_metric = trainer.evaluate_loader(train_loader)
    test_loss, test_metric = trainer.evaluate_loader(test_loader)

    logging.info("+" * 50)
    logging.info(f"Task ID: {task_id}")
    logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
    logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")
    logging.info("+" * 50)

    return train_loss

def optimize_model(args, train_loader, test_loader, task_id):
    """
    Optimize the hyperparameters for the client model using Optuna.
    Args:
        args:
        train_loader(torch.utils.data.DataLoader):  Training data loader.
        test_loader(torch.utils.data.DataLoader):  Testing data loader.
        task_id(str):  Task ID.

    Returns:
        dict:  Dictionary containing the trajectory of the model checkpoints.

    """
    abs_log_dir = os.path.abspath(args.logs_dir)
    storage_name = f"sqlite:////{abs_log_dir}/hp_dashboard_{task_id}.db"
    if args.test_best_hyperparams is True:
        study = optuna.load_study(storage=storage_name)
    else:
        study = optuna.create_study(direction="minimize",
                                    storage=storage_name,
                                    load_if_exists=True)
        study.optimize(lambda trial: objective(trial=trial, train_loader=train_loader, test_loader=test_loader, task_id=task_id, args=args),
                       n_trials=args.n_trials)

    best_params = study.best_params
    logging.info("=" * 100)
    logging.info(f"Best hyperparameters: {study.best_params}")
    logging.info(f"Optimization results saved in: hp_dashboard_{task_id}.db")
    logging.info("=" * 100)

    trainer = initialize_trainer(args, learning_rate=best_params['lr'],
                                 weight_decay=best_params['weight_decay'],
                                 beta1=best_params['beta1'],
                                 beta2=best_params['beta2'])

    trajectory_dict = dict()

    for step in range(args.num_rounds):
        trainer.fit_epoch(train_loader)
        if step % args.save_freq == 0:
            os.makedirs(os.path.join(args.local_models_dir, task_id), exist_ok=True)

            path = os.path.join(args.local_models_dir, task_id, f"{step}.pt")
            path = os.path.abspath(path)
            trainer.save_checkpoint(path)
            trajectory_dict[f"{step}"] = path

        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()

    train_loss, train_metric = trainer.evaluate_loader(train_loader)
    test_loss, test_metric = trainer.evaluate_loader(test_loader)

    logging.info("+" * 50)
    logging.info(f"Task ID: {task_id}")
    logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
    logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")
    logging.info("+" * 50)

    local_model_path = os.path.join(args.local_models_dir, task_id, f"{args.num_rounds - 1}.pt")
    local_model_path = os.path.abspath(local_model_path)

    return trajectory_dict



def main():
    """
    Train local models using the federated dataset.
    Returns:
        None
    """

    args = parse_args()

    rng = np.random.default_rng(seed=args.seed)

    set_seeds(args.seed)

    configure_logging(args)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)

    if args.hparams_config_path is None:
        raise ValueError("Hyperparameters configuration file is not provided.")

    if not os.path.exists(args.hparams_config_path):
        raise FileNotFoundError(f"Hyperparameters configuration file not found at '{args.hparams_config_path}'.")

    logging.info("=" * 100)
    logging.info("Launch hyperparameter optimization using Optuna..")

    models_trajectory_dict = dict()

    for task_id in tqdm(federated_dataset.task_id_to_name):
        train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        abs_log_dir = os.path.abspath(args.logs_dir)
        os.makedirs(abs_log_dir, exist_ok=True)

        task_trajectory_dict = optimize_model(args, train_loader, test_loader, task_id)
        models_trajectory_dict[task_id] = task_trajectory_dict

    local_models_trajectory_path = os.path.join(args.metadata_dir, "local_trajectories.json")
    with open(local_models_trajectory_path, "w") as f:
        json.dump(models_trajectory_dict, f)

    logging.info(f'The metadata dictionary has been saved in {local_models_trajectory_path}')

if __name__ == "__main__":
    main()


