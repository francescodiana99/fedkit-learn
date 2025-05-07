"""
Active server isolation simulation script.

The script simulates the scenario where an active server isolates a client by sending back the same exact model that he receives,
 forcing a local training. The isolation process is described in (https://arxiv.org/abs/2108.06910)."""

import argparse
import os
import logging

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from fedklearn.trainer.trainer import Trainer, DPTrainer

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from fedklearn.metrics import multiclass_accuracy

from tqdm import tqdm

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
        "--metadata_dir",
        type=str,
        help="Metadata directory",
        required=True
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Number of additional epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Logs directory"
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=1,
        help="Logging frequency"
    )
    parser.add_argument(
        "--iso_chkpts_dir",
        type=str,
        default="./isolated_models",
        help="Directory to save isolated models."
    )
    parser.add_argument(
        "--attacked_round",
        required=True,
        type=int,
        help="Starting round for attacking."
    )
    parser.add_argument(
        '--attacked_task',
        type=str,
        help="If set, only the specified task will be attacked.",
        default=None
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Saving frequency."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
    "--quiet",
    action="store_true",
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)
    
def write_logs(logger, loss, metric, counter, mode="train", epsilon=None):
    """Log training and testing metrics using the provided logger."""

    if mode == "train":
        logger.add_scalar("Train/Loss", loss, counter)
        logger.add_scalar("Train/Metric", metric, counter)
        logging.info(f"Train Loss: {loss:.4f} | Train Metric: {metric:.4f}")
    elif mode == "test":
        logger.add_scalar("Test/Loss", loss, counter)
        logger.add_scalar("Test/Metric", metric, counter)
        logging.info(f"Test Loss: {loss:.4f} | Test Metric: {metric:.4f}")
    else:
        raise ValueError(f"Invalid mode '{mode}'. Accepted modes are 'train' and 'test'.")
    
    if epsilon is not None:
        logger.add_scalar("Train/Epsilon", epsilon, counter)
        logging.info(f"Epsilon: {epsilon:.4f}")

def initialize_attack_trainer(args, fl_setup, client_messages_metadata, train_loader=None):
    """
    Initialize the trainer for running the active part of the simulation.

    Args:
        args (argparse.Namespace): Command-line arguments.
        fl_setup (dict): Federated learning setup dictionary.
        client_messages_metadata (dict): Metadata of the client messages.
        train_loader (torch.utils.data.DataLoader, optional): Training data loader. Defaults to None.
    Returns:
        Trainer: Trainer object
    """
    
    local_model_chkpt = torch.load(client_messages_metadata['local'][f"{args.attacked_round}"],
                                    map_location=args.device,
                                    weights_only=False)["model_state_dict"]
    
    criterion, metric, cast_float = get_trainers_config(fl_setup["task_name"])
    criterion.to(args.device)
    attacked_model = initialize_model(fl_setup["model_config_path"])
    attacked_model.load_state_dict(local_model_chkpt)

    if fl_setup["optimizer"] == "sgd":

        optimizer_params = {
            "lr": fl_setup["lr"],
            "momentum": fl_setup["momentum"],
            "weight_decay": fl_setup["weight_decay"],
            "init_fn": optim.SGD
        }

        optimizer = optim.SGD(
            [param for param in attacked_model.parameters() if param.requires_grad],
            lr=fl_setup["lr"],
            momentum=fl_setup["momentum"],
            weight_decay=fl_setup["weight_decay"]
        )
    elif fl_setup["optimizer"] == "adam":
        optimizer = optim.Adam(
            [param for param in attacked_model.parameters() if param.requires_grad],
            lr=fl_setup["lr"],
            weight_decay=fl_setup["weight_decay"]
        )
        optimizer_params = {
            "lr": fl_setup["lr"],
            "weight_decay": fl_setup["weight_decay"],
            "init_fn": optim.Adam
        }
    else:
        raise NotImplementedError(
            f"Optimizer '{fl_setup["optimizer"]}' is not implemented"
        )

    if train_loader is not None:
        if fl_setup["num_active_rounds"] != args.num_rounds:
            raise ValueError(
                f"Number of  expected active rounds {fl_setup['num_active_rounds']} is not equal to the number of rounds set {args.num_rounds}. \
                Please set the same number of rounds in order to correctly compute the privacy budget."
                )
        trainer = DPTrainer(
            model=attacked_model,
            optimizer=optimizer,
            criterion=criterion,
            device=args.device,
            cast_float=cast_float,
            metric=metric,
            max_physical_batch_size=fl_setup["max_physical_batch_size"] if fl_setup["max_physical_batch_size"] else fl_setup["batch_size"],
            noise_multiplier=fl_setup["noise_multiplier"],
            clip_norm=fl_setup["clip_norm"],
            epsilon=fl_setup["dp_epsilon"],
            delta=fl_setup["dp_delta"],
            epochs=(args.num_rounds + args.attacked_round + 1) * fl_setup["local_steps"], # +1 needed beacause round starts from 0, e.g. to get 150 we need 100 + 49 (50th round attacked) +1
            train_loader=train_loader,
            optimizer_init_dict=optimizer_params,
            rng=torch.Generator(device=args.device).manual_seed(args.seed)
        )
        trainer.load_checkpoint(client_messages_metadata['local'][f"{args.attacked_round}"])
        return trainer

    else:
        return Trainer(
            model=attacked_model,
            optimizer=optimizer,
            criterion=criterion,
            device=args.device,
            cast_float=cast_float,
            metric=metric
        )

def main():
    args = parse_args()

    set_seeds(args.seed)

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)

    try:
        with open(os.path.join(args.metadata_dir, "setup.json"), "r") as f:
            fl_setup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Federated Learning simulation metadata file not found at \
                                '{args.metadata_dir}/setup.json'.")
    
    use_dp = True if fl_setup["clip_norm"] is not None else False

    federated_dataset = load_dataset(fl_setup, rng=rng)

    num_clients = len(federated_dataset.task_id_to_name)

    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        all_messages_metadata = json.load(f)

    logging.info("Simulate Attacks..")

    os.makedirs(args.iso_chkpts_dir, exist_ok=True)

    all_isolated_models_metadata_dict = defaultdict(lambda : dict())
    final_isolated_models_metadata_dict = defaultdict(lambda : dict())

    pbar = tqdm(range(num_clients))
    compute_single_client = False
    if args.attacked_task is not None:
        attacked_client_id = int(federated_dataset.task_id_to_name[args.attacked_task])
        compute_single_client = True
    else:
        attacked_client_id = 0

    while attacked_client_id < num_clients:
        logging.info("=" * 100)
        logging.info(f"Isolating client {attacked_client_id}")

        train_dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=fl_setup["batch_size"], shuffle=True)

        test_dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=fl_setup["batch_size"], shuffle=False)

        client_messages_metadata = {
            "global": all_messages_metadata["global"],
            "local": all_messages_metadata[f"{attacked_client_id}"]
        }

        if use_dp:
            active_trainer = initialize_attack_trainer(args=args, fl_setup=fl_setup, client_messages_metadata=client_messages_metadata,
                                                        train_loader=train_loader)
        else:
            active_trainer = initialize_attack_trainer(args=args, fl_setup=fl_setup, client_messages_metadata=client_messages_metadata)
            
        logger = SummaryWriter(log_dir=os.path.join(args.logs_dir, f"{attacked_client_id}"))
        if not fl_setup["by_epoch"]:
            if use_dp:
                raise NotImplementedError("Simulating local steps with DP is not supported. Only local epochs are currently supported.")
            if fl_setup["local_steps"] is None:
                raise ValueError('Please specify a number of local steps to simulate.')
            train_iterator = iter(train_loader)
        
        counter = 0

        for step in range(args.num_rounds):
            if fl_setup["by_epoch"]:
                if use_dp:
                    train_loss, train_metric, epsilon = active_trainer.fit_epoch()
                else:
                    train_loss, train_metric = active_trainer.fit_epoch(loader=train_loader)
            else:
                for _ in range(fl_setup["local_steps"]):
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        batch = next(train_iterator)

                    train_loss, train_metric = active_trainer.fit_batch(batch)
            
            if step % args.log_freq == 0:
                logging.info("+" * 50)
                test_loss, test_metric = active_trainer.evaluate_loader(loader=test_loader)
                if use_dp:
                    write_logs(logger, train_loss, train_metric, counter, mode="train", epsilon=epsilon)
                    write_logs(logger, test_loss, test_metric, counter, mode="test")
                
                else:
                    write_logs(logger, train_loss, train_metric, counter, mode="train")
                    write_logs(logger, test_loss, test_metric, counter, mode="test")

            if step % args.save_freq == 0:
                os.makedirs(os.path.join(args.iso_chkpts_dir, f"{attacked_client_id}"), exist_ok=True)
                chkpt_path = os.path.abspath(os.path.join(os.path.join(args.iso_chkpts_dir, f"{attacked_client_id}", f"{step}.pt")))
                active_trainer.save_checkpoint(chkpt_path)
                all_isolated_models_metadata_dict[f"{attacked_client_id}"][f"{step}"] = chkpt_path

                if active_trainer.lr_scheduler is not None:
                    active_trainer.lr_scheduler.step()

            counter += 1

        last_saved_iteration = max(all_isolated_models_metadata_dict[f"{attacked_client_id}"], key=int)

        final_isolated_models_metadata_dict[f"{attacked_client_id}"] = (
            all_isolated_models_metadata_dict[f"{attacked_client_id}"][last_saved_iteration]
        )

        logging.info("Local model isolated successfully.")

        attacked_client_id += 1
        pbar.update(1)
        if compute_single_client:
            attacked_client_id = num_clients

    pbar.close()

    all_isolated_models_metadata_dict = swap_dict_levels(all_isolated_models_metadata_dict)
    trajectory_path = os.path.join(args.metadata_dir, f"isolated_trajectories_{args.attacked_round}.json")
    with open(trajectory_path, "w") as f:
        json.dump(all_isolated_models_metadata_dict, f)

    logging.info(f"The attacked models have been successfully saved in {trajectory_path} .")

    logging.info("="*100)
    logging.info("Saving final models metadata...")


    with open(os.path.join(args.metadata_dir, f"last_isolated_{args.attacked_round}.json"), "w") as f:
        json.dump(final_isolated_models_metadata_dict, f)

    logging.info("The final isolated models have been successfully saved.")

if __name__ == "__main__":
    main()
