"""
Local model's optimization simulation script.

The script simulates the scenario where each client optimizes its local model without any communication with the server. Local models are used as oracle to test
the performance of the model-based attribute inference attack."""

import argparse
import logging
import os
import pathlib
import optuna
import numpy as np
from datetime import datetime


from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedklearn.trainer.trainer import Trainer, DPTrainer
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
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer"
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
        "--local_chkpts_dir",
        type=str,
        default="./local_models",
        help="Directory to save local models checkpoints"
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

    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Task to optimize. If set, only the specified task will be optimized."
    )


    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def initialize_trainer(args, simulation_setup, learning_rate, weight_decay, beta1, beta2, use_dp=False, train_loader=None):
    """
    Initialize the trainer based on the specified task.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        simulation_setup (dict): Simulation setup dictionary.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        beta1 (float): Beta1 for Adam optimizer.
        beta2 (float): Beta2 for Adam optimizer.
        use_dp (bool): Flag for using differential privacy.
        train_loader (torch.utils.data.DataLoader): Training data loader.

    Returns:
        Trainer: Initialized trainer.
    """

    model = initialize_model(simulation_setup["model_config_path"])

    task_config = {
        "adult": (nn.BCEWithLogitsLoss(), binary_accuracy_with_sigmoid, True),
        "toy_classification": (nn.BCEWithLogitsLoss(), binary_accuracy_with_sigmoid, True),
        "toy_regression": (nn.MSELoss(), mean_squared_error, False),
        "purchase": (nn.CrossEntropyLoss(), multiclass_accuracy, False),
        "purchase_binary": (nn.BCEWithLogitsLoss(), binary_accuracy_with_sigmoid, True),
        "medical_cost": (nn.MSELoss(), mean_absolute_error, False),
        "linear_medical_cost": (nn.MSELoss(), mean_absolute_error, False),
        "income": (nn.MSELoss(), mean_absolute_error, False),
        "binary_income": (nn.BCEWithLogitsLoss(), binary_accuracy_with_sigmoid, True),
        "linear_income": (nn.MSELoss(), mean_absolute_error, False),
    }

    if simulation_setup["task_name"] not in task_config.keys():
        raise ValueError(f"Task name '{simulation_setup['task_name']}' is not supported.")
    criterion, metric, cast_float = task_config[simulation_setup["task_name"]]

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        optimizer_params = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "init_fn": optim.SGD
        }

    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
        optimizer_params = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "init_fn": optim.Adam,
            "betas": (beta1, beta2)
        }
    else:
        raise NotImplementedError(
            f"Optimizer '{args.optimizer}' is not supported."
        )

    if use_dp:
        return DPTrainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=args.device,
            optimizer=optimizer,
            cast_float=cast_float,
            max_physical_batch_size=simulation_setup["max_physical_batch_size"] \
                if simulation_setup["max_physical_batch_size"] is not None else simulation_setup["batch_size"],
            noise_multiplier=simulation_setup["noise_multiplier"],
            epsilon=simulation_setup["dp_epsilon"],
            delta=simulation_setup["dp_delta"],
            clip_norm=simulation_setup["clip_norm"],
            epochs=args.num_rounds,
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


def objective(trial, simulation_setup, train_loader, test_loader, task_id, args):
    """
    Initialize the objective function for the hyperparameter optimization using Optuna.
    For additional details, please refer to the Optuna documentation:
    https://optuna.readthedocs.io/en/stable/index.html

    Args:
        trial (optuna.Trial): Optuna trial object.
        simulation_setup (dict): Simulation setup dictionary.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        test_loader (torch.utils.data.DataLoader): Testing data loader.
        task_id (str): Task ID to optimize.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        float: Objective value.
     """
    use_dp = simulation_setup["clip_norm"] is not None
    with open(args.hparams_config_path, "r") as f:
        hparams_dict = json.load(f)
    beta1 = trial.suggest_float("beta1", hparams_dict['beta1'][0], hparams_dict['beta1'][1])
    beta2 = trial.suggest_float("beta2", hparams_dict['beta2'][0], hparams_dict['beta2'][1])
    lr = trial.suggest_float("lr", hparams_dict['lr'][0], hparams_dict['lr'][1], log=True)
    weight_decay = trial.suggest_float("weight_decay", hparams_dict['weight_decay'][0], hparams_dict['weight_decay'][1],
                                       log=True)

    trainer = initialize_trainer(args, simulation_setup=simulation_setup, learning_rate=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                                 use_dp=use_dp, train_loader=train_loader)

    for _ in range(args.num_rounds):
        if use_dp:
            _ ,_ , epsilon = trainer.fit_epoch()
        else:
            trainer.fit_epoch(train_loader)
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()

    train_loss, train_metric = trainer.evaluate_loader(train_loader)
    test_loss, test_metric = trainer.evaluate_loader(test_loader)

    logging.info("+" * 50)
    logging.info(f"Task ID: {task_id}")
    logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
    logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")
    if use_dp:
        logging.info(f"Epsilon: {epsilon:.4f}")
    logging.info("+" * 50)

    return train_loss

def write_logs(train_loss, train_metric, test_loss, test_metric, step, logger):
    """
    Write the training and testing logs to the TensorBoard.
    Args:
        args:
        train_loss(float):  Training loss.
        train_metric(float):  Training metric.
        test_loss(float):  Testing loss.
        test_metric(float):  Testing metric.
        step(int): Current step.
        logger(torch.utils.tensorboard.SummaryWriter):  Logger object.

    Returns:
        None
    """
    logger.add_scalar("Train/Loss", train_loss, step)
    logger.add_scalar("Train/Metric", train_metric, step)
    logger.add_scalar("Test/Loss", test_loss, step)
    logger.add_scalar("Test/Metric", test_metric, step)

    logging.info("+" * 50)
    logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f}|")
    logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")

def optimize_model(args, simulation_setup, train_loader, test_loader, task_id, logs_dir):
    """
    Optimize the hyperparameters for the local model using Optuna.
    Args:
        args (argparse.Namespace):  Parsed command-line arguments.
        simulation_setup (dict):  Simulation setup dictionary.
        train_loader(torch.utils.data.DataLoader):  Training data loader.
        test_loader(torch.utils.data.DataLoader):  Testing data loader.
        task_id(str): Task ID.
        logs_dir(str): Directory to save the logs.

    Returns:
        trajectory_dict (dict):  Dictionary containing the trajectory of the model checkpoints.
        setup_dict (dict):  Dictionary containing the hyperparameters setup of the simulation.

    """
    abs_log_dir = os.path.abspath(args.logs_dir)
    storage_name = f"sqlite:////{abs_log_dir}/hp_dashboard_{task_id}.db"

    # TODO: check if this works as it should
    if args.test_best_hyperparams is True:
        logging.info(f'Loading best hyperparameters from {os.path.join(args.metadata_dir, "local_setup.json")}')
        with open(os.path.join(args.metadata_dir, "local_setup.json"), "r") as f:
            setup_dict = json.load(f)
        if task_id not in setup_dict.keys():
            raise ValueError(f"Task ID {task_id} not found in the setup dictionary.")
        best_params = setup_dict[task_id]
        simulation_setup["batch_size"] = best_params["batch_size"]
        args.num_rounds = best_params["num_rounds"]

    else:
        study = optuna.create_study(direction="minimize",
                                    storage=storage_name,
                                    load_if_exists=True,
                                    study_name=f"{datetime.now()}")
        study.optimize(lambda trial: objective(trial=trial, simulation_setup=simulation_setup, train_loader=train_loader, test_loader=test_loader, task_id=task_id, args=args),
                       n_trials=args.n_trials)

        best_params = study.best_params
        logging.info(f"Optimization results saved in: hp_dashboard_{task_id}.db")
    logging.info("=" * 100)
    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Optimization results saved in: hp_dashboard_{task_id}.db")
    logging.info("=" * 100)

    if simulation_setup["clip_norm"] is not None:
        trainer = initialize_trainer(args, simulation_setup, learning_rate=best_params['lr'],
                                     weight_decay=best_params['weight_decay'],
                                     beta1=best_params['beta1'],
                                     beta2=best_params['beta2'],
                                     use_dp=True,
                                     train_loader=train_loader)
    else:
        trainer = initialize_trainer(args, simulation_setup, learning_rate=best_params['lr'],
                                    weight_decay=best_params['weight_decay'],
                                    beta1=best_params['beta1'],
                                    beta2=best_params['beta2'])

    trajectory_dict = dict()

    logger = SummaryWriter(os.path.join(logs_dir, task_id))

    setup_dict = {
            "beta1": best_params['beta1'],
            "beta2": best_params['beta2'],
            "lr": best_params['lr'],
            "weight_decay": best_params['weight_decay'],
            "num_rounds": args.num_rounds,
            "local_chkpts_dir": os.path.join(logs_dir, task_id),
            "batch_size": simulation_setup["batch_size"],
            "n_trials": args.n_trials
        }

    for step in range(args.num_rounds):
        if simulation_setup["clip_norm"] is not None:
            train_loss, train_metric, epsilon = trainer.fit_epoch()
        else:
            train_loss, train_metric = trainer.fit_epoch(train_loader)
        if step % args.save_freq == 0:
            os.makedirs(os.path.join(args.local_chkpts_dir, task_id), exist_ok=True)

            path = os.path.join(args.local_chkpts_dir, task_id, f"{step}.pt")
            path = os.path.abspath(path)
            trainer.save_checkpoint(path)
            trajectory_dict[f"{step}"] = path

        if step % args.log_freq == 0:
            test_loss, test_metric = trainer.evaluate_loader(test_loader)
            write_logs(train_loss, train_metric, test_loss, test_metric, step, logger)

        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()

    if simulation_setup["clip_norm"] is not None:
        train_loss, train_metric = trainer.evaluate_loader(train_loader)
    else:
        test_loss, test_metric = trainer.evaluate_loader(test_loader)

    logging.info("+" * 50)
    logging.info(f"Task ID: {task_id}")
    logging.info(f"Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")
    logging.info(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f} |")
    if simulation_setup["clip_norm"] is not None:
        logging.info(f"Epsilon: {epsilon:.4f}")
    logging.info("+" * 50)

    return trajectory_dict, setup_dict


def main():
    """
    Train local models to simulate the empirical optimal model for each client.
    Returns:
        None
    """

    args = parse_args()

    rng = np.random.default_rng(seed=args.seed)

    set_seeds(args.seed)

    configure_logging(args)
    
    try:
        with open(os.path.join(args.metadata_dir, "setup.json"), "r") as f:
            simulation_setup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Federated Learning simulation metadata file not found at \
                                '{args.metadata_dir}/setup.json'. Ensure to run the simulation script first.")

    use_dp = simulation_setup["clip_norm"] is not None

    federated_dataset = load_dataset(simulation_setup, rng=rng)

    if args.hparams_config_path is None:
        raise ValueError("Hyperparameters configuration file is not provided.")
    if not os.path.exists(args.hparams_config_path):
        raise FileNotFoundError(f"Hyperparameters configuration file not found at '{args.hparams_config_path}'.")

    logging.info("=" * 100)
    logging.info("Launch hyperparameter optimization using Optuna..")

    models_trajectory_dict = dict()
    tasks_setup_dict = dict()

    if args.task_id is None:
        for task_id in tqdm(federated_dataset.task_id_to_name):
            train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
            test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

            train_loader = DataLoader(train_dataset, batch_size=simulation_setup["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=simulation_setup["batch_size"], shuffle=False)

            abs_log_dir = os.path.abspath(args.logs_dir)
            os.makedirs(abs_log_dir, exist_ok=True)

            task_trajectory_dict, setup_dict = optimize_model(args, simulation_setup, train_loader, test_loader, task_id, abs_log_dir)
            models_trajectory_dict[task_id] = task_trajectory_dict
            tasks_setup_dict[task_id] = setup_dict
    else:

        train_dataset = federated_dataset.get_task_dataset(args.task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(args.task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=simulation_setup["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=simulation_setup["batch_size"], shuffle=False)

        abs_log_dir = os.path.abspath(args.logs_dir)
        os.makedirs(abs_log_dir, exist_ok=True)

        task_trajectory_dict, setup_dict = optimize_model(args, simulation_setup, train_loader, test_loader, args.task_id, abs_log_dir)
        models_trajectory_dict[args.task_id] = task_trajectory_dict
        tasks_setup_dict[args.task_id] = setup_dict


    local_models_trajectory_path = os.path.join(args.metadata_dir, "local_trajectories.json")
    setup_dict_path = os.path.join(args.metadata_dir, "local_setup.json")
    os.makedirs(args.metadata_dir, exist_ok=True)
    with open(local_models_trajectory_path, "w") as f:
        json.dump(models_trajectory_dict, f)
    with open(setup_dict_path, "w") as f:
        json.dump(tasks_setup_dict, f)

    logging.info(f'The metadata trajectory dictionary has been saved in {local_models_trajectory_path}')
    logging.info(f'The metadata setup dictionary has been saved in {setup_dict_path}')

if __name__ == "__main__":
    main()
