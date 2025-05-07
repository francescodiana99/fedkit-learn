"""
Active server optimization simulation script.

The script simulates the scenario where an active server isolates a client and performs an additional virtual update using Adam Optimization Algorithm.
The implementation corresponds to Algortitm 3 in the paper.
"""

import argparse
import datetime
import logging
import os

from datetime import datetime

import numpy as np

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedklearn.trainer.trainer import Trainer, DPTrainer
from fedklearn.federated.client import Client, DPClient
from fedklearn.federated.simulator import ActiveAdamFederatedAveraging

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

    # Federated learning args
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
        "--active_chkpts_dir",
        type=str,
        default=None,
        help="Checkpoints directory for the active models"
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
        "--beta1",
        type=float,
        default=None,
        help="Beta1 parameter for the Adam optimizer in the active server"
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=None,
        help="Beta2 parameter for the Adam optimizer in the active server"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon stability constant for the Adam optimizer in the active server"
    )

    parser.add_argument(
        "--attacked_round",
        type=int,
        default=None,
        help="Starting round of the active attack"
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

    parser.add_argument(
        "--use_norm",
        action="store_true",
        help="If set, the optimization minimizes the norm of the pseudo-gradients, as proxy of the loss"
    )

    parser.add_argument(
        "--attacked_task",
        type=str,
        default=None,
        help="If set, the active attack will be performed on the specified client. Default is None"
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

def save_active_setup(args, setup_dict):
    """
    Save the active attack setup to the metadata directory.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        setup_dict (dict): Dictionary containing the active attack setup hyperparameters.
    """
    os.makedirs(args.metadata_dir, exist_ok=True)
    with open(os.path.join(args.metadata_dir, f"active_setup_round_{args.attacked_round}.json"), "w") as f:
        json.dump(setup_dict, f)

def save_last_round_metadata(all_messages_metadata, metadata_dir, args):
    """"
    Save the last round metadata to the metadata directory.
    Args:
        all_messages_metadata (dict): Dictionary containing the messages metadata.
        metadata_dir (str): Path to the metadata directory.
        args (argparse.Namespace): Parsed command-line arguments.
    """
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

    if args.use_norm:
        last_models_metadata_path = os.path.join(metadata_dir, f"last_active_{args.attacked_round}_norm.json")
    else:
        last_models_metadata_path = os.path.join(metadata_dir, f"last_active_{args.attacked_round}.json")
    last_models_dict = read_dict(last_models_metadata_path)
    last_models_dict[f'{args.attacked_round}'] = last_clients_messages_metadata
    with open(last_models_metadata_path, "w") as f:
        json.dump(last_models_dict, f)

    logging.info(f"Last models sent by the client saved to {last_models_metadata_path}")

def objective(trial, federated_dataset, fl_setup, rng, args):
    """
    Initialize the objective function for the hyperparameter optimization using Optuna.
    For additional details, please refer to the Optuna documentation:
    https://optuna.readthedocs.io/en/stable/index.html

    Args:
        trial (optuna.Trial): Optuna trial object.
        federated_dataset (FederatedDataset): Federated dataset for the specific task.
        fl_setup (dict): Dictionary containing the federated learning setup hyperparameters.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        float: Objective value.
     """
    with open(args.hparams_config_path, "r") as f:
        hparams_dict = json.load(f)
    beta1 = trial.suggest_float("beta1", hparams_dict['beta1'][0], hparams_dict['beta1'][1])
    beta2 = trial.suggest_float("beta2", hparams_dict['beta2'][0], hparams_dict['beta2'][1])
    alpha = trial.suggest_float("alpha", hparams_dict['alpha'][0], hparams_dict['alpha'][1], log=True)

    clients = load_clients_from_chkpt(federated_dataset, args, fl_setup)
    simulator = initialize_active_simulator(clients=clients, fl_setup=fl_setup, rng=rng, beta1=beta1, beta2=beta2, alpha=alpha, args=args)

    logging.info(f"Running active simulation with beta1={beta1}, beta2={beta2}, alpha={alpha}..")
    logging.info('Initial logs..')
    simulator.write_logs(display_only=True)
    for round_id in tqdm(range(args.num_rounds)):
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


def initialize_trainer(args, models_metadata_dict, fl_setup, task_id=None, mode='global', train_loader=None):
    """
    Initialize the trainer based on the specified model metadata.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        models_metadata_dict (dict): Dictionary containing the model metadata.
        fl_setup (dict): Dictionary containing the federated learning setup hyperpamrameters.
        task_id (str, optional): Task ID to load. Default is None.
        mode (str, optional): Mode for the initialization. Default is 'global'.
        train_loader (torch.utils.data.DataLoader, optional): Training data loader. Default is None.

    Returns:
        Trainer: Initialized trainer.
    """

    use_dp = True if fl_setup["clip_norm"] is not None else False

    model = initialize_model(fl_setup["model_config_path"])

    criterion, metric, cast_float = get_trainers_config(fl_setup["task_name"])
    criterion = criterion.to(args.device)

    if fl_setup["optimizer"] == "sgd":
        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=fl_setup["lr"],
            momentum=fl_setup["momentum"],
            weight_decay=fl_setup["weight_decay"],
        )

        optimizer_params = {
            "lr": fl_setup["lr"],
            "momentum": fl_setup["momentum"],
            "weight_decay": fl_setup["weight_decay"],
            "init_fn": optim.SGD
        }

    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=fl_setup["lr"],
            weight_decay=fl_setup["weight_decay"],
        )

        optimizer_params = {
            "lr": fl_setup["lr"],
            "weight_decay": fl_setup["weight_decay"],
            "init_fn": optim.Adam,
            "betas": (0.99, 0.999)
        }

    else:
        raise NotImplementedError(
            f"Optimizer '{fl_setup["optimizer"]}' is not implemented."
        )
    if mode == 'task':
        model_chkpt_path = models_metadata_dict[task_id][f'{args.attacked_round}']
    elif mode == 'global':
        model_chkpt_path = models_metadata_dict[f'{args.attacked_round}']
    else:
        raise ValueError(f"Mode '{mode}' is not recognized.")
    model_chkpts = torch.load(model_chkpt_path, map_location=args.device, weights_only=False)["model_state_dict"]
    model.load_state_dict(model_chkpts)

    if mode == 'task' and use_dp:
        trainer = DPTrainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=args.device,
            optimizer=optimizer,
            cast_float=cast_float,
            max_physical_batch_size=fl_setup["max_physical_batch_size"] if fl_setup["max_physical_batch_size"]  is not None else fl_setup["batch_size"],
            noise_multiplier=fl_setup["noise_multiplier"],
            epsilon=fl_setup["dp_epsilon"],
            delta=fl_setup["dp_delta"],
            clip_norm=fl_setup["clip_norm"],
            epochs=(args.num_rounds + args.attacked_round + 1) * fl_setup["local_steps"],
            train_loader=train_loader,
            optimizer_init_dict=optimizer_params,
            rng=torch.Generator(device=args.device).manual_seed(args.seed)
        )
        trainer.load_checkpoint(model_chkpt_path)
    else:
        trainer =  Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            metric=metric,
            device=args.device,
            cast_float=cast_float
            )

    return trainer

def initialize_active_simulator(args, clients, fl_setup, rng, beta1, beta2, alpha):
    """
    Initialize the simulation with an active server, based on the specified arguments.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        clients (list): List of clients.
        fl_setup (dict): Dictionary containing the federated learning setup hyper
        rng (numpy.random.Generator): Random number generator.
        beta1 (float): Beta1 parameter for the Adam optimizer.
        beta2 (float): Beta2 parameter for the Adam optimizer.
        alpha (float): Learning rate parameter for the Adam optimizer.
    """
    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        model_metadata_dict = json.load(f)

    if f"{args.attacked_round}" not in model_metadata_dict['global'].keys():
        raise ValueError(f"Round {args.attacked_round} not found in the global model metadata.")

    global_trainer = initialize_trainer(args, model_metadata_dict['global'], fl_setup, task_id=None, mode='global')
    global_logger = SummaryWriter(os.path.join(args.logs_dir, "global"))

    use_dp = True if fl_setup["clip_norm"] is not None else False

    simulator = ActiveAdamFederatedAveraging(
        clients=clients,
        global_trainer=global_trainer,
        logger=global_logger,
        chkpts_dir=fl_setup["chkpts_dir"],
        rng=rng,
        beta1=beta1,
        beta2=beta2,
        epsilon=1e-8,
        alpha=alpha,
        attacked_round=args.attacked_round,
        active_chkpts_dir=args.active_chkpts_dir,
        use_dp=use_dp
    )

    return simulator


def load_clients_from_chkpt(federated_dataset, args, fl_setup):
    """
    Load clients from given checkpoints.

    Args:
        federated_dataset (FederatedDataset): Federated dataset for the specific task.
        args (argparse.Namespace): Parsed command-line arguments.
        fl_setup (dict): Dictionary containing the federated learning setup hyperparameters.
    Returns:
        list: List of clients.
    """

    use_dp = True if fl_setup["clip_norm"] is not None else False

    set_seeds(args.seed)
    clients = []
    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        models_metadata_dict = json.load(f)
    if args.attacked_task is None:
        for task_id in federated_dataset.task_id_to_name:

            train_dataset = federated_dataset.get_task_dataset(task_id, mode="train")
            test_dataset = federated_dataset.get_task_dataset(task_id, mode="test")

            train_loader = DataLoader(train_dataset, batch_size=fl_setup["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=fl_setup["batch_size"], shuffle=False)

            logger = SummaryWriter(os.path.join(args.logs_dir, f"{task_id}"))
            if use_dp:
                trainer = initialize_trainer(args, models_metadata_dict, fl_setup, task_id=task_id, mode='task', 
                                            train_loader=train_loader)
                client = DPClient(trainer=trainer,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            local_steps=fl_setup["local_steps"],
                            by_epoch=fl_setup["by_epoch"],
                            logger=logger,
                            name=args.attacked_task
                            )
            else:
                trainer = initialize_trainer(args, models_metadata_dict, fl_setup, task_id=task_id, mode='task')
                client = Client(trainer=trainer,
                                train_loader=train_loader,
                                test_loader=test_loader,
                                local_steps=fl_setup["local_steps"],
                                by_epoch=fl_setup["by_epoch"],
                                logger=logger,
                                name=task_id
                                )
            clients.append(client)
    else:

        train_dataset = federated_dataset.get_task_dataset(args.attacked_task, mode="train")
        test_dataset = federated_dataset.get_task_dataset(args.attacked_task, mode="test")

        train_loader = DataLoader(train_dataset, batch_size=fl_setup["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=fl_setup["batch_size"], shuffle=False)

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{args.attacked_task}"))
        if use_dp:
            if args.num_rounds != fl_setup["num_active_rounds"]:
                raise ValueError(f"Number of rounds {args.num_rounds} does not match the expected number of active rounds {fl_setup['num_active_rounds']}.")
            trainer = initialize_trainer(args, models_metadata_dict, fl_setup, task_id=args.attacked_task, mode='task', train_loader=train_loader)
            client = DPClient(trainer=trainer,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            local_steps=fl_setup["local_steps"],
                            by_epoch=fl_setup["by_epoch"],
                            logger=logger,
                            name=args.attacked_task
                            )
        else:
            trainer = initialize_trainer(args, models_metadata_dict, fl_setup, task_id=args.attacked_task, mode='task')
            client = Client(trainer=trainer,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        local_steps=fl_setup["local_steps"],
                        by_epoch=fl_setup["by_epoch"],
                        logger=logger,
                        name=args.attacked_task
                        )
        clients.append(client)
    return clients

def main():
    """
    Execute the active isolation part of a federated learning simulation. Implements Algorithm 3 of the paper.

    This function initializes the federated dataset, clients, and simulator based on the
    provided command-line arguments. It then runs the simulation for the specified number
    of rounds, saving model checkpoints and logs at specified intervals. Finally, it saves
    the messages' metadata.

    Returns:
        None
    """

    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)
    set_seeds(args.seed)

    configure_logging(args)

    try:
        with open(os.path.join(args.metadata_dir, "setup.json"), "r") as f:
            fl_setup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Federated Learning simulation metadata file not found at \
                                '{args.metadata_dir}/setup.json'.")

    federated_dataset = load_dataset(fl_setup=fl_setup, rng=rng)

    if args.optimize_hyperparams:
        if args.hparams_config_path is None:
            raise ValueError("Hyperparameters configuration file is not provided.")
        if not os.path.exists(args.hparams_config_path):
            raise FileNotFoundError(f"Hyperparameters configuration file not found at '{args.hparams_config_path}'.")

        logging.info("=" * 100)
        logging.info("Launch hyperparameter optimization using Optuna..")
        abs_log_dir = os.path.abspath(args.logs_dir)
        os.makedirs(abs_log_dir, exist_ok=True)
        storage_name = f"sqlite:////{abs_log_dir}/hp_dashboard_{args.attacked_round}.db"

        study = optuna.create_study(direction="minimize",
                                    storage=storage_name,
                                    load_if_exists=True, study_name=f'{datetime.now()}')
        study.optimize(lambda trial: objective(trial, federated_dataset, fl_setup, rng, args), 
                       n_trials=args.n_trials)

        best_params = study.best_params

        logging.info("=" * 100)
        logging.info(f"Best hyperparameters: {study.best_params}")
        logging.info(f"Optimization results saved in: {abs_log_dir}/hp_dashboard_{args.attacked_round}.db")
        logging.info("=" * 100)

    logging.info("Loading clients from checkpoints...")

    clients = load_clients_from_chkpt(federated_dataset=federated_dataset, args=args, fl_setup=fl_setup)

    logging.info("=" * 100)
    logging.info("Initializing simulator from checkpoint..")

    if args.optimize_hyperparams:
        simulator = initialize_active_simulator(clients=clients,
                                                rng=rng,
                                                fl_setup=fl_setup,
                                                beta1=best_params['beta1'],
                                                beta2=best_params['beta2'], 
                                                alpha=best_params['alpha'], 
                                                args=args)
        
        logging.info(f"Running active simulation with hyperparameters beta1={best_params['beta1']},\
                      beta2={best_params['beta2']}, alpha={best_params['alpha']}...")
        setup_dict = {
            "beta1": best_params['beta1'],
            "beta2": best_params['beta2'],
            "alpha": best_params['alpha'],
            "use_norm": args.use_norm,
            "num_rounds": args.num_rounds,
            "active_chkpts_dir": simulator.active_chkpts_dir
        }

    else:
        simulator = initialize_active_simulator(clients=clients, 
                                                rng=rng, 
                                                fl_setup=fl_setup, 
                                                beta1=args.beta1, 
                                                beta2=args.beta2,
                                                alpha=args.alpha, 
                                                args=args)

        logging.info(f"Running active simulation with hyperparameters beta1={args.beta1}, beta2={args.beta2}, \
                      alpha={args.alpha}...")
        setup_dict = {
            "beta1": args.beta1,
            "beta2": args.beta2,
            "alpha": args.alpha,
            "use_norm": args.use_norm,
            "num_rounds": args.num_rounds,
            "active_chkpts_dir": simulator.active_chkpts_dir
        }

    logging.info("=" * 100)
    logging.info("Write initial logs..")
    simulator.write_logs()

    logging.info("=" * 100)
    logging.info('Run active simulation...')

    for round_id in tqdm(range(args.num_rounds)):
        logs_flag = (round_id % args.log_freq == 0)
        chkpts_flag = (round_id % args.save_freq == 0)

        simulator.simulate_active_round(save_chkpts=chkpts_flag, save_logs=logs_flag)
    logging.info('Last Results..')
    simulator.write_logs(display_only=True)

    logging.info("=" * 100)
    logging.info("Saving simulation results..")
    os.makedirs(os.path.dirname(args.metadata_dir), exist_ok=True)
    if args.use_norm:
        messages_metadata_path = os.path.join(args.metadata_dir, f"active_trajectories_norm_{args.attacked_round}.json")
    else:
        messages_metadata_path = os.path.join(args.metadata_dir, f"active_trajectories_{args.attacked_round}.json")
    with open(messages_metadata_path, "w") as f:
        json.dump(simulator.messages_metadata, f)

    logging.info(f"The messages metadata dictionary has been saved in {messages_metadata_path}")

    save_last_round_metadata(simulator.messages_metadata, args.metadata_dir, args)
    save_active_setup(args, setup_dict)

if __name__ == "__main__":
    main()
