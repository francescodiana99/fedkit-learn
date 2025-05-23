"Reconstruction and Evaluation of Linear Models script "

import os
from fedklearn.models.linear import LinearLayer
from fedklearn.trainer.trainer import Trainer
from fedklearn.metrics import binary_accuracy_with_sigmoid, mean_squared_error, mean_absolute_error
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random

from utils import *

import numpy as np
import json
from fedklearn.utils import get_param_tensor, set_param_tensor
from fedklearn.metrics import mean_squared_error, mean_absolute_error, threshold_binary_accuracy
import random
import argparse
import logging



def parse_args():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--metadata_dir',
        type=str,
        default='./metadata',
        help='Task metadata directory'
        )

    parser.add_argument(
        '--reconstructed_models_dir',
        type=str,
        help='Reconstructed models directory. If None, the models will not be saved.',
        default=None
        )


    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
        )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='PyTorch device'
        )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=10_000,
        help='Number of trials for getting Theta_out'
        )


    parser.add_argument(
        '--sensitive_attribute',
        type=str,
        help='Sensitive attribute to attack'
        )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Results directory'
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

    return parser.parse_args()

def initialize_trainer(model, criterion, metric, cast_float, device):
    """
    Initialize the trainer based on the specified model metadata.
    Args:
        model (torch.nn): PyTorch model.
        args(argparse.Namespace): Parsed command-line arguments.
    Returns:
        Trainer: Initialized trainer.
    """

    optimizer = optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=0.1,
        momentum=0,
        weight_decay=0,
    )

    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        device=device,
        cast_float=cast_float
        )


def reconstruct_model_params(args, metadata_dict, task_id):
    """
    Analytical reconstruction of a linear model.

    Args:
        args(argparse.Namespace): Arguments for the reconstruction.
        metadata_dict(dict): Dictionary containing messages metadata.
        task_id(str): task id of the attacked client

    Returns:
        reconstructed_params(torch.Tensor): Tensor containing the reconstructed model parameters
    """

    messages_metadata = get_messages_metadata(metadata_dict, task_id=task_id)

    round_ids = list(messages_metadata["global"].keys())

    pseudo_grads = compute_pseudo_grads(messages_metadata, round_ids)

    global_params_list = [get_param_tensor(get_model_at_round(messages_metadata, i, mode='global')) for i in round_ids]

    theta_in = torch.stack(global_params_list).numpy()
    theta_out = np.c_[pseudo_grads.numpy(), np.ones(pseudo_grads.shape[0])]

    recon_rounds = test_rounds(round_ids, theta_in.shape[1] + 1, args.n_trials, theta_out)

    Theta_out_cal = theta_out[recon_rounds]
    Theta_in_cal = theta_in[recon_rounds]
    result = np.linalg.inv(Theta_out_cal) @ Theta_in_cal
    return torch.tensor(result[-1]).float()


def test_rounds(rounds_list, num_rounds, n_trials, theta_out):
    """
    Test the condition number of the matrix formed by the pseudo-gradients and a column of ones for different rounds.
    Args:
        rounds_list(list): List of round identifiers.
        num_rounds(int): Number of rounds to consider.
        n_trials(int): Number of trials to perform.
        theta_out(np.ndarray): Matrix containing the pseudo-gradients and a column of ones.
    Returns:
        best_round(list): List of round identifiers that minimize the condition number of the matrix.
    """
    rounds = extract_rounds(rounds_list, num_rounds, n_trials)
    best_cond_num = np.inf
    for round_ind in rounds:
        round_ind = [int(i) for i in round_ind]
        round_ind.sort()
        Theta_out_cal = theta_out[round_ind]
        cond = np.linalg.cond(Theta_out_cal)
        if cond < best_cond_num:
            best_cond_num = cond
            best_round = round_ind
    logging.info(f'Best condition number: {np.linalg.cond(theta_out[best_round])}')
    logging.info(f'Best round: {best_round}')
    return best_round

def extract_rounds(rounds_list, num_rounds, n_trials):
    """
    Extract a set of communication rounds.
    Args:
        rounds_list(list): List of round identifiers.
        num_rounds(int): Number of rounds to consider.
        n_trials(int): Number of trials to perform.
    Returns:
        rounds(list): List of communication rounds.
    """
    all_possible_rounds = set()
    rounds = []

    while len(rounds) < n_trials:
        round_set = tuple(sorted(random.sample(rounds_list, num_rounds)))
        if round_set not in all_possible_rounds:
            all_possible_rounds.add(round_set)
            rounds.append(round_set)

    return rounds


def get_messages_metadata(all_messages_metadata_dict, task_id):
    """
    Extract task messages metadata from a dict.
    Args:
        all_messages_metadata_dict(dict): Dictionary containing all messages metadata.
        task_id(str): task id of the attacked client

    Returns:
        messages_metadata(dict): Dictionary containing messages metadata for the task
    """
    messages_metadata = {"global": all_messages_metadata_dict["global"],
                         "local": all_messages_metadata_dict[f"{task_id}"]}

    return messages_metadata


def get_model_at_round(messages_metadata, round_id, mode="global"):
    """
    Retrieve the model at a specific communication round.

    Parameters:
    - round_id (int): The identifier of the communication round.
    - mode (str): The mode specifying whether to retrieve a local or global model (default is "global").

    Returns:
    - torch.nn.Module: The model at the specified communication round.
    """

    assert mode in {"local", "global"}, f"`mode` should be 'local' or 'global', not {mode}"
    model_chkpts = torch.load(messages_metadata[mode][round_id], weights_only=False)["model_state_dict"]
    model = LinearLayer(input_dimension=model_chkpts['fc.weight'].shape[1], output_dimension=1, bias=True)
    model.load_state_dict(model_chkpts)

    return model


def compute_pseudo_gradient_at_round(messages_metadata, round_id):
    """
    Compute the pseudo-gradient at a specific communication round.
    Args:
        messages_metadata(dict): Dictionary containing messages metadata.
        round_id(int): The identifier of the communication round.
    Returns:
        pseudo_gradient(torch.Tensor): The pseudo-gradient at the specified communication round.
    """
    global_model = get_model_at_round(messages_metadata, round_id=round_id, mode="global")
    local_model = get_model_at_round(messages_metadata, round_id=round_id, mode="local")

    global_param_tensor = get_param_tensor(global_model)
    local_param_tensor = get_param_tensor(local_model)

    return global_param_tensor - local_param_tensor


def compute_pseudo_grads(messages_metadata, round_ids):
    """
    Compute the pseudo-gradients at multiple communication rounds.
    Args:
        messages_metadata(dict): Dictionary containing messages metadata.
        round_ids(list): List of communication round identifiers.
    Returns:
        torch.Tensor: The pseudo-gradients at the specified communication rounds."""
    pseudo_gradients_dict = dict()
    for round_id in round_ids:
        pseudo_gradients_dict[round_id] = compute_pseudo_gradient_at_round(messages_metadata, round_id=round_id)

    return torch.stack(list(pseudo_gradients_dict.values()), dim=0)

def main():

    args = parse_args()

    configure_logging(args)

    set_seeds(args.seed)

    rng = np.random.default_rng(args.seed)

    try:
        with open(os.path.join(args.metadata_dir, "setup.json"), "r") as f:
            fl_setup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Federated Learning simulation metadata file not found at \
                                '{args.metadata_dir}/setup.json'.")
    
    with open(os.path.join(args.metadata_dir, 'federated.json'), 'r') as f:
        metadata_dict = json.load(f)

    with open(os.path.join(args.metadata_dir, 'local_trajectories.json'), 'r') as f:
        local_models_dict = json.load(f)

    criterion, metric, cast_float = get_trainers_config(fl_setup["task_name"]) 
    model_init_fn = lambda: initialize_model(fl_setup["model_config_path"])
    
    federated_dataset = load_dataset(fl_setup, rng=rng)

    results_dict = {'attack_accuracy': dict(),
                    'norm_diff': dict(),
                    'loss_diff': dict(),
                    'n_samples': dict(),
                    'reconstructed_loss': dict(),
                    'optimal_loss': dict(),
                    'device': get_device_info()
                    }


    for task_id in tqdm(federated_dataset.task_id_to_name):
        logging.info('+' * 50)
        logging.info(f'Reconstructing task {task_id}..')

        reconstructed_params = reconstruct_model_params(args=args, metadata_dict=metadata_dict, task_id=task_id)

        recon_model = model_init_fn()
        set_param_tensor(model=recon_model, param_tensor=reconstructed_params, device=args.device)

        emp_opt_model = model_init_fn()
        last_round = max([int(i) for i in local_models_dict[f'{task_id}'].keys()])
        emp_opt_chkpts = torch.load(local_models_dict[f'{task_id}'][f'{last_round}'], weights_only=False)["model_state_dict"]
        emp_opt_model.load_state_dict(emp_opt_chkpts)

        train_dataset = federated_dataset.get_task_dataset(task_id, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)

        recon_trainer = initialize_trainer(recon_model, criterion, metric, cast_float, args.device)
        emp_opt_trainer = initialize_trainer(emp_opt_model, criterion, metric, cast_float, args.device)

        recon_loss, recon_metric = recon_trainer.evaluate_loader(train_loader)
        emp_opt_loss, emp_opt_metric = emp_opt_trainer.evaluate_loader(train_loader)

        logging.info(f'Reconstructed Loss: {recon_loss} | Metric: {recon_metric}')
        logging.info(f'Local Loss: {emp_opt_loss} | Metric: {emp_opt_metric}')

        sensitive_attribute_id = train_dataset.column_name_to_id[args.sensitive_attribute]

        recon_aia_score = evaluate_mb_aia(
            model=recon_model,
            dataset=train_dataset,
            sensitive_attribute_id=sensitive_attribute_id,
            sensitive_attribute_type='binary',
            initialization='normal',
            device=args.device,
            criterion=criterion,
            cast_float=cast_float,
            learning_rate=1,
            num_iterations=100,
            optimizer_name='sgd',
            success_metric=threshold_binary_accuracy
        )

        opt_aia_score = evaluate_mb_aia(
                model=emp_opt_model,
                dataset=train_dataset,
                sensitive_attribute_id=sensitive_attribute_id,
                sensitive_attribute_type='binary',
                initialization='normal',
                device=args.device,
                criterion=criterion,
                cast_float=cast_float,
                learning_rate=1,
                num_iterations=100,
                optimizer_name='sgd',
                success_metric=threshold_binary_accuracy
            )

        logging.info(f'Reconstructed AIA Score: {recon_aia_score} | Optimal AIA Score: {opt_aia_score}')
        logging.info('+' * 50)

        emp_opt_params = emp_opt_trainer.get_param_tensor().cpu().numpy()
        norm_distance = np.linalg.norm(emp_opt_params - reconstructed_params.numpy())

        logging.info(f'Norm distance: {norm_distance}')
        logging.info(f'Loss difference: {abs(emp_opt_loss - recon_loss)}')

        if args.reconstructed_models_dir is not None:
            os.makedirs(args.reconstructed_models_dir, exist_ok=True)
            checkpoint = {'model_state_dict': recon_model.state_dict()}
            torch.save(checkpoint, os.path.join(args.reconstructed_models_dir, f'{task_id}.pt'))

        results_dict['attack_accuracy'][f'{task_id}'] = str(recon_aia_score)
        results_dict['attack_accuracy'][f'{task_id}'] = str(opt_aia_score)
        results_dict['norm_diff'][f'{task_id}'] = str(norm_distance)
        results_dict['loss_diff'][f'{task_id}'] = str(abs((emp_opt_loss - recon_loss)))
        results_dict['n_samples'][f'{task_id}'] = str(len(train_dataset))
        results_dict['reconstructed_loss'][f'{task_id}'] = str(recon_loss)
        results_dict['optimal_loss'][f'{task_id}'] = str(emp_opt_loss)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'linear_reconstruction.json'), 'w') as f:
        json.dump(results_dict, f)

    logging.info(f'Results correctly saved in {os.path.join(args.results_dir, 'linear_reconstruction.json')}')


if __name__ == '__main__':
    main()
