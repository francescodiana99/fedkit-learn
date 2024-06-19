import argparse
import logging
import os
import pathlib
import shutil

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

import optuna

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
                 'income'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification', 'purchase', 'medical_cost',"
             " 'income'.",
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
        default=0,
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
        "--is_binary_classification",
        action="store_true",
        help="Flag for binary classification",
        default=False
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
        help="Mixing coefficient for the mixing sample distribution in Adult and Income dataset"
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

    parser.add_argument(
        "--beta1",
        type=float,
        default=None,
        help="Beta1 parameter for the Adam optimizer in the active attack scenario"
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=None,
        help="Beta2 parameter for the Adam optimizer in the active attack scenario"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon parameter for the Adam optimizer in the active attack scenario"
    )

    parser.add_argument(
        "--attacked_round",
        type=int,
        default=None,
        help="Round in which the active attack is performed"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha parameter for the Adam optimizer in the active attack scenario"
    )

    parser.add_argument(
        "--optimize_hyperparams",
        action="store_true",
        help="Flag for optimizing hyperparameters, if True, the simulator will run the optimization using Optuna"
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


    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)

# TODO: save for each server model the metadata
def save_last_round_metadata(all_messages_metadata, metadata_dir, args):
    last_saved_round_id = max(map(int, all_messages_metadata["0"].keys()))
    _random_key = list(all_messages_metadata["0"].keys())[0]
    last_saved_round_id = int(last_saved_round_id) if isinstance(_random_key, int) else str(last_saved_round_id)

    last_clients_messages_metadata = dict()
    # last_server_messages_metadata = dict()

    for client_id in all_messages_metadata:
        if str(client_id).isdigit():
            last_clients_messages_metadata[client_id] = all_messages_metadata[client_id][last_saved_round_id]
            # last_server_messages_metadata[client_id] = all_messages_metadata["global"][last_saved_round_id]

    last_models_metadata_path = os.path.join(metadata_dir, "last_active.json")
    last_models_dict = read_dict(last_models_metadata_path)
    last_models_dict[f'{args.attacked_round}'] = last_clients_messages_metadata
    with open(last_models_metadata_path, "w") as f:
        json.dump(last_models_dict, f)

    logging.info(f"Last models sent by the client saved to {last_models_metadata_path}")

def objective(trial , federated_dataset, rng, args):
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
    alpha = trial.suggest_float("alpha", hparams_dict['alpha'][0], hparams_dict['alpha'][1], log=True)

    clients = load_clients_from_chkpt(federated_dataset, args)
    simulator = initialize_active_simulator(clients, rng, args)

    simulator.beta1 = beta1
    simulator.beta2 = beta2
    simulator.alpha = alpha

    for round_id in range(args.attacked_round):
        simulator.simulate_round(save_chkpts=False, save_logs=False)

    train_loss = simulator.get_client_avg_train_loss()

    return train_loss


def initialize_trainer(models_metadata_dict, task_id, args, mode='global'):
    """
    Initialize the trainer based on the specified model metadata.
    Args:
        model_metadata_dict (dict): Dictionary containing model's trajectory metadata.
        task_id (str): Task identifier.
        args(argparse.Namespace): Parsed command-line arguments.
        mode (str): Mode of the trainer. It can be 'global' or 'task'.
    Returns:
        Trainer: Initialized trainer.
    """

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
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{args.optimizer}' is not implemented."
        )
    if mode == 'task':
        model_chkpt_path = models_metadata_dict[task_id][f'{args.attacked_round}']
    elif mode == 'global':
        model_chkpt_path = models_metadata_dict[f'{args.attacked_round}']
    else:
        raise ValueError(f"Mode '{mode}' is not recognized.")
    model_chkpts = torch.load(model_chkpt_path, map_location=args.device)["model_state_dict"]
    model.load_state_dict(model_chkpts)

    return Trainer(
        model=model,
                   criterion=criterion,
                   optimizer=optimizer,
                   metric=metric,
                   device=args.device
                   )


def initialize_active_simulator(clients, rng, args):

    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        model_metadata_dict = json.load(f)

    global_trainer = initialize_trainer(model_metadata_dict['global'], args.attacked_round, args, mode='global')
    global_logger = SummaryWriter(os.path.join(args.logs_dir, "global"))

    active_chkpts_dir = os.path.join(args.chkpts_dir)
    simulator = ActiveAdamFederatedAveraging(
        clients=clients,
        global_trainer=global_trainer,
        logger=global_logger,
        chkpts_dir=active_chkpts_dir,
        rng=rng,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        alpha=args.alpha,
        attacked_round=args.attacked_round
    )

    return simulator


def load_clients_from_chkpt(federated_dataset, args):
    """
    Load clients from checkpoints.

    Args:

    Returns:
        list: List of clients.
    """
    clients = []
    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        models_metadata_dict = json.load(f)
    for task_id in federated_dataset.task_id_to_name:

        trainer = initialize_trainer(models_metadata_dict, task_id, args, mode='task')

        train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
        test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{task_id}"))

        client = Client(trainer=trainer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        local_steps=args.local_steps,
                        by_epoch=args.by_epoch,
                        logger=logger,
                        )
        clients.append(client)
    return clients

def main():
    """
    Execute the active part of a federated learning simulation.

    This function initializes the federated dataset, clients, and simulator based on the
    provided command-line arguments. It then runs the simulation for the specified number
    of rounds, saving model checkpoints and logs at specified intervals. Finally, it saves
    the messages' metadata.

    Returns:
        None
    """

    args = parse_args()

    rng = np.random.default_rng(seed=args.seed)

    configure_logging(args)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)

    if args.optimize_hyperparams:
        if args.hparams_config_path is None:
            raise ValueError("Hyperparameters configuration file is not provided.")

        if not os.path.exists(args.hparams_config_path):
            raise FileNotFoundError(f"Hyperparameters configuration file not found at '{args.hparams_config_path}'.")

        logging.info("=" * 100)
        logging.info("Launch hyperparameter optimization using Optuna..")

        storage_name = f"sqlite:///{args.logs_dir}/hp_dashboard.db"

        study = optuna.create_study(direction="minimize", storage=storage_name, load_if_exists=True)
        study.optimize(lambda trial: objective(trial, federated_dataset, rng, args), n_trials=args.n_trials)

        best_params = study.best_params

        logging.info("=" * 100)
        logging.info(f"Best hyperparameters: {study.best_params}")
        logging.info(f"Optimization results saved in: {args.log_dir}/hp_dashboard.db")
        logging.info("=" * 100)

    logging.info("Loading clients from checkpoints...")

    clients = load_clients_from_chkpt(federated_dataset, args)

    logging.info("=" * 100)
    logging.info("Initializing simulator from checkpoint..")
    simulator = initialize_active_simulator(clients, rng, args)

    if args.optimize_hyperparams:
        simulator.beta1 = best_params['beta1']
        simulator.beta2 = best_params['beta2']
        simulator.alpha = best_params['alpha']

        logging.info("Running active simulation with the best hyperparameters...")

    logging.info("=" * 100)
    logging.info("Write initial logs..")
    simulator.write_logs()

    logging.info("=" * 100)
    logging.info('Run active simulation...')

    for round_id in tqdm(range(args.num_rounds)):
        logs_flag = (round_id % args.log_freq == 0)
        chkpts_flag = (round_id % args.save_freq == 0)

        simulator.simulate_active_round(save_chkpts=chkpts_flag, save_logs=logs_flag)

    logging.info("=" * 100)
    logging.info("Saving simulation results..")
    os.makedirs(os.path.dirname(args.metadata_dir), exist_ok=True)
    messages_metadata_path = os.path.join(args.metadata_dir, f"active_{args.attacked_round}.json")
    with open(messages_metadata_path, "w") as f:
        json.dump(simulator.messages_metadata, f)

    logging.info(f"The messages metadata dictionary has been saved in {messages_metadata_path}")

    save_last_round_metadata(simulator.messages_metadata, args.metadata_dir, args)


if __name__ == "__main__":
    main()
