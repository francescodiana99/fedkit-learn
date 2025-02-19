import copy
import os
import json
import logging
from collections import defaultdict
import sys

from tqdm import tqdm

import numpy as np
import torch
import random

from fedklearn.datasets.income.income import FederatedIncomeDataset
from fedklearn.datasets.purchase.purchase import FederatedPurchaseDataset, FederatedPurchaseBinaryClassificationDataset
from fedklearn.datasets.medical_cost.medical_cost import FederatedMedicalCostDataset
from fedklearn.metrics import *

from fedklearn.models.linear import LinearLayer, TwoLinearLayers
from fedklearn.models.sequential import SequentialNet
from fedklearn.trainer.trainer import Trainer, DebugTrainer

from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.datasets.toy.toy import FederatedToyDataset

from fedklearn.attacks.aia import ModelDrivenAttributeInferenceAttack
from fedklearn.attacks.sia import SourceInferenceAttack


def set_seeds(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (int): The random seed to be set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    """
    Seed the worker for reproducibility.
    Args:
        worker_id:

    Returns:

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def none_or_float(value):
    """
    Helper function to convert 'none' to None or float values.
    """
    if value.lower() == 'none':
        return None
    else:
        return float(value)


def swap_dict_levels(nested_dict):
    """
    Swap the levels of keys in a nested dictionary.

    Parameters:
    - nested_dict (dict): The nested dictionary where the first level represents outer keys
                         and the second level represents inner keys.

    Returns:
    - dict: A new dictionary with swapped levels, where the first level represents inner keys
            and the second level represents outer keys.
    """
    swapped_dict = {}

    for outer_key, inner_dict in nested_dict.items():
        for inner_key, data in inner_dict.items():
            if inner_key not in swapped_dict:
                swapped_dict[inner_key] = {}
            swapped_dict[inner_key][outer_key] = data

    return swapped_dict


def configure_logging(args):
    """
    Set up logging based on verbosity level
    """
    # TODO: this should be fixed. Opacus changes the default level to WARNING
    logging.basicConfig(level=logging.INFO - (args.verbose - args.quiet) * 10)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.setLevel(logging.INFO - (args.verbose - args.quiet) * 10)

# TODO: check where to use
def get_task_type(task_name):
    task_types = {
        "adult": "binary_classification",
        "toy_classification": "binary_classification",
        "toy_regression": "regression",
        "purchase": "classification",
        "purchase_binary": "binary_classification",
        "medical_cost": "regression",
        "income": "regression",
        "linear_income": "regression",
        "linear_medical_cost": "regression"

    }
    if task_name not in task_types.keys():
        raise NotImplementedError(
            f"Network initialization for task '{task_name}' is not implemented"
        )

    return task_types[task_name]

def load_dataset(fl_setup, rng):
    """
    Load a federated dataset based on the specified task name.

    Args:
        fl_setup (dict): Dictionary containing the setup for the federated learning experiment.
        rng (RandomState): NumPy random number generator for reproducibility.

    Returns:
        FederatedDataset: Initialized federated dataset.

    Raises:
        NotImplementedError: If the dataset initialization for the specified task is not implemented.
    """

    task_name = fl_setup["task_name"]
    data_path = fl_setup["data_path"]
    scale_target = fl_setup["scale_target"]

    # TODO: if have time, fix adult or remove it
    if task_name == "adult":
        with open(data_path, "r") as f:
            metadata_dict = json.load(f)
        split_criterion = metadata_dict["split_criterion"]

        if split_criterion is None:
            raise ValueError("Split criterion must be specified for the Adult dataset.")
        if split_criterion == "n_tasks":
            n_tasks = metadata_dict["n_tasks"]
            n_task_samples = metadata_dict["n_task_samples"]
            return FederatedAdultDataset(
                cache_dir=data_dir,
                download=False,
                rng=rng,
                split_criterion=split_criterion,
                n_tasks=n_tasks,
                n_task_samples=n_task_samples,
            )
        elif split_criterion == "correlation" or split_criterion == 'flip':
            mixing_coefficient = metadata_dict["mixing_coefficient"]
            return FederatedAdultDataset(
                cache_dir=data_dir,
                download=False,
                rng=rng,
                split_criterion=split_criterion,
                mixing_coefficient=mixing_coefficient,
            )
        else:
            return FederatedAdultDataset(
                cache_dir=data_dir,
                download=False,
                rng=rng,
                split_criterion=split_criterion,
            )
    elif task_name == "income":
        with open(data_path, "r") as f:
            metadata_dict = json.load(f)
        split_criterion = metadata_dict["split_criterion"]
        n_tasks = metadata_dict["n_tasks"]
        n_task_samples = metadata_dict["n_task_samples"]
        cache_dir = metadata_dict['cache_dir']
        state= metadata_dict['state']
        mixing_coefficient = metadata_dict['mixing_coefficient']
        return FederatedIncomeDataset(
            cache_dir=cache_dir,
            download=False,
            split_criterion=split_criterion,
            mixing_coefficient=mixing_coefficient,
            state=state,
            n_tasks=n_tasks,
            n_task_samples=n_task_samples,
            scale_target=scale_target
        )
    elif task_name == "binary_income":
        with open(data_path, "r") as f:
            metadata_dict = json.load(f)
        split_criterion = metadata_dict["split_criterion"]
        n_tasks = metadata_dict["n_tasks"]
        n_task_samples = metadata_dict["n_task_samples"]
        cache_dir = metadata_dict['cache_dir']
        state= metadata_dict['state']
        mixing_coefficient = metadata_dict['mixing_coefficient']
        return FederatedIncomeDataset(
            cache_dir=cache_dir,
            download=False,
            split_criterion=split_criterion,
            mixing_coefficient=mixing_coefficient,
            state=state,
            n_tasks=n_tasks,
            n_task_samples=n_task_samples,
            binarize=True
        )
    # elif task_name == "linear_income":
    #     with open(data_path, "r") as f:
    #         metadata_dict = json.load(f)
    #     split_criterion = metadata_dict["split_criterion"]
    #     n_tasks = metadata_dict["n_tasks"]
    #     n_task_samples = metadata_dict["n_task_samples"]
    #     cache_dir = metadata_dict['cache_dir']
    #     state= metadata_dict['state']
    #     mixing_coefficient = metadata_dict['mixing_coefficient']
    #     return FederatedIncomeDataset(
    #         cache_dir=cache_dir,
    #         download=False,
    #         split_criterion=split_criterion,
    #         mixing_coefficient=mixing_coefficient,
    #         state=state,
    #         n_tasks=n_tasks,
    #         n_task_samples=n_task_samples,
    #         use_linear=True
    #     )
    
    # TODO: fix if have time
    elif task_name == "purchase":
        with open(os.path.join(data_dir, "split_criterion.json"), "r") as f:
            split_dict = json.load(f)
        split_criterion = split_dict["split_criterion"]
        n_tasks = split_dict["n_tasks"]
        n_task_samples = split_dict["n_task_samples"]
        return FederatedPurchaseDataset(
            cache_dir=data_dir,
            download=False,
            force_generation=False,
            rng=rng,
            split_criterion=split_criterion,
            n_tasks=n_tasks,
            n_task_samples=n_task_samples
        )

    elif task_name == "purchase_binary":
        with open(os.path.join(data_dir, "split_criterion.json"), "r") as f:
            split_dict = json.load(f)
        split_criterion = split_dict["split_criterion"]
        test_frac = split_dict["test_frac"]
        return FederatedPurchaseBinaryClassificationDataset(
            cache_dir=data_dir,
            download=False,
            force_generation=False,
            rng=rng,
            split_criterion=split_criterion,
            test_frac=test_frac,
            target_item='130'
        )

    elif task_name == "toy_regression" or task_name == "toy_classification":
        with open(data_path, "r") as f:
            metadata_dict = json.load(f)
        cache_dir = metadata_dict['cache_dir']
        return FederatedToyDataset(
            cache_dir=cache_dir,
            allow_generation=False,
            force_generation=False,
            rng=rng
        )
    elif task_name == "medical_cost":
        with open(data_path, "r") as f:
            metadata_dict = json.load(f)
        split_criterion = metadata_dict["split_criterion"]
        cache_dir = metadata_dict['cache_dir']
        n_tasks = metadata_dict["n_tasks"]
        return FederatedMedicalCostDataset(
            cache_dir=cache_dir,
            download=False,
            force_generation=False,
            rng=rng,
            split_criterion=split_criterion,
            n_tasks=n_tasks,
            scale_target=scale_target
        )

    # elif task_name == "linear_medical_cost":
    #     with open(data_path, "r") as f:
    #         metadata_dict = json.load(f)
    #     split_criterion = metadata_dict["split_criterion"]
    #     cache_dir = metadata_dict['cache_dir']
    #     n_tasks = metadata_dict["n_tasks"]
    #     return FederatedMedicalCostDataset(
    #         cache_dir=cache_dir,
    #         download=False,
    #         force_generation=False,
    #         rng=rng,
    #         split_criterion=split_criterion,
    #         n_tasks=n_tasks,
    #         use_linear=True
    #     )
    else:
        raise NotImplementedError(
            f"Dataset initialization for task '{task_name}' is not implemented."
        )


def get_last_rounds(round_ids, keep_frac=0.):
    """
    Extracts a subset of the given round_ids, starting from a specific index.

    Parameters:
    - round_ids (list of str): A list of round identifiers.
    - keep_frac (float, optional): Fraction of rounds to keep.
      If set to 0.0, all rounds, except the last, will be discarded.
      If set to 1.0, all rounds will be kept.
      If set to a value between 0.0 and 1.0, it determines the fraction of rounds to keep
      starting from the end of the list. Defaults to 0. (i.e., discarding all rounds, except the last).

    Returns:
    - set of str: A subset of round_ids based on the specified keep_frac.

    Example:
    >>> round_ids_list = ['1', '2', '3', '4', '5']
    >>> get_last_rounds(round_ids_list, keep_frac=0.2)
    {'4', '5'}

    Note:
    - The round_ids are assumed to be sortable as strings.
    - The function ensures that at least one round is kept, even when keep_frac is 0.
    """
    assert 0 <= keep_frac <= 1, "keep_frac must be in the range (0, 1)"

    n_rounds = len(round_ids)
    start_index = int((n_rounds - 1) * (1. - keep_frac))

    int_list = sorted(list(map(int, round_ids)))

    return set(map(str, int_list[start_index:]))

def get_first_rounds(round_ids, keep_frac=0.):
    """
    Extracts a subset of the given round_ids, starting from a specific index.

    Parameters:
    - round_ids (list of str): A list of round identifiers.
    - keep_frac (float, optional): Fraction of rounds to keep.
      If set to 0.0, all rounds, except the first, will be discarded.
      If set to 1.0, all rounds will be kept.
      If set to a value between 0.0 and 1.0, it determines the fraction of rounds to keep
      starting from the beginning of the list. Defaults to 0. (i.e., discarding all rounds, except the first).

    Returns:
    - list of str: A subset of round_ids based on the specified keep_frac.

    Example:
    >>> round_ids_list = ['1', '2', '3', '4', '5']
    >>> get_first_rounds(round_ids_list, keep_frac=0.2)
    {'1', '2'}

        Note:
        - The round_ids are assumed to be sortable as strings.
        - The function ensures that at least one round is kept, even when keep_frac is 0.
        """
    assert 0 <= keep_frac <= 1, "keep_frac must be in the range (0, 1)"

    n_rounds = len(round_ids)
    end_index = int((n_rounds - 1) * keep_frac) + 1

    int_list = sorted(list(map(int, round_ids)))

    return set(map(str, int_list[:end_index]))

def get_trainer_parameters(task_name, device, model_config_path):
    # TODO: absolutely needs to be refactored
    model_init_fn = lambda: initialize_model(model_config_path)
    if task_name == "adult":
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
        metric = binary_accuracy_with_sigmoid
        is_binary_classification = True
    elif task_name == "toy_classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
        is_binary_classification = True
        metric = binary_accuracy_with_sigmoid
    elif task_name == "toy_regression":
        criterion = nn.MSELoss(reduction="none").to(device)
        is_binary_classification = False
        metric = mean_squared_error
    elif task_name == "purchase":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        is_binary_classification = False
        metric = multiclass_accuracy
    elif task_name == "purchase_binary":
        criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
        is_binary_classification = True
        metric = binary_accuracy_with_sigmoid
    elif task_name == "medical_cost":
        criterion = nn.MSELoss(reduction="none").to(device)
        is_binary_classification = True
        metric = mean_squared_error

    elif task_name == "income":
        criterion = nn.MSELoss(reduction="none").to(device)
        is_binary_classification = True
        metric = mean_squared_error
    elif task_name == "linear_income":
        criterion = nn.MSELoss(reduction="none").to(device)
        is_binary_classification = True
        metric = mean_squared_error
    elif task_name == "linear_medical_cost":
        criterion = nn.MSELoss(reduction="none").to(device)
        is_binary_classification = True
        metric = mean_squared_error


    else:
        raise NotImplementedError(
            f"Network initialization for task '{task_name}' is not implemented"
        )

    return criterion, model_init_fn, is_binary_classification, metric


def initialize_trainers_dict(models_metadata_dict, criterion, model_init_fn, is_binary_classification, metric, device):

    """
    Initialize trainers for models based on the provided dictionary mapping IDs to model paths.

    Parameters:
    - models_metadata_dict (Dict[str: str]): A dictionary mapping model IDs to their corresponding paths.
    - criterion (torch.nn.Module) : The loss function.
    - model_init_fn (Callable): The function used to initialize the models.
    - is_binary_classification (bool): Indicates whether the task is binary classification.
    - device (str): The device (e.g., 'cpu' or 'cuda') on which the models will be initialized.

    Returns:
    - Dict[str: Trainer]: A dictionary mapping model IDs to initialized trainers for these models.
    """
    optimizer = None

    trainers_dict = dict()
    for client_id in models_metadata_dict:
        model_chkpts = torch.load(models_metadata_dict[client_id], map_location=device)["model_state_dict"]
        model = model_init_fn()
        model.load_state_dict(model_chkpts)

        trainers_dict[client_id] = Trainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            is_binary_classification=is_binary_classification
        )

    return trainers_dict


def evaluate_mb_aia(
        model, dataset, sensitive_attribute_id, sensitive_attribute_type, initialization, device, num_iterations,
        criterion, is_binary_classification, learning_rate, optimizer_name, success_metric, rng=None, torch_rng=None,
        output_losses=False, output_predictions=False
):
    """
    Evaluate the Model-based Attribute Inference Attack on a given model and dataset.
    
    Parameters:
    - model (torch.nn.Module): The target model to test.
    - dataset (FederatedDataset): The dataset to use for the attack.
    - sensitive_attribute_id (int): The index of the sensitive attribute in the dataset.
    - sensitive_attribute_type (str): The type of the sensitive attribute.
    - initialization (str): The initialization method for reconstructing continuous sensitive attributes.
    - device (str): The device on which the attack will be executed
    - num_iterations (int): The number of iterations for the optimization part when attacking continuous features.
    - criterion (torch.nn.Module): The loss function to use for the attack.
    - is_binary_classification (bool): Indicates whether the task is binary classification.
    - learning_rate (float): The learning rate for the optimization part when attacking continuous features.
    - optimizer_name (str): The name of the optimizer to use for the optimization part when attacking continuous features.
    - success_metric (str): The metric to use for evaluating the success of the attack.
    - rng (np.random.RandomState): The random number generator for reproducibility.
    - torch_rng (torch.Generator): The PyTorch random number generator for reproducibility.
    - output_losses (bool): Whether to output the losses during the attack.
    - output_predictions (bool): Whether to output the predictions during the attack.

    Returns:
    - score (float): The score of the attack
    """

    attack_simulator = ModelDrivenAttributeInferenceAttack(
        model=model,
        dataset=dataset,
        sensitive_attribute_id=sensitive_attribute_id,
        sensitive_attribute_type=sensitive_attribute_type,
        initialization=initialization,
        device=device,
        criterion=criterion,
        is_binary_classification=is_binary_classification,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        success_metric=success_metric,
        rng=rng,
        torch_rng=torch_rng
    )

    all_losses = attack_simulator.execute_attack(num_iterations=num_iterations, output_loss=True)
    score = attack_simulator.evaluate_attack()

    # Output losses and predictions for debugging purposes
    if output_losses:
        logging.info(f"{all_losses[:20]}")

    if output_predictions:
        logging.info(f"{all_losses[:20].argmax(axis=1)}")
    return float(score)

def weighted_average(scores, n_samples):
    if len(scores) != len(n_samples):
        raise ValueError("The lengths of 'scores' and 'n_samples' must be the same.")

    weighted_sum = sum(score * n_sample for score, n_sample in zip(scores, n_samples))

    total_samples = sum(n_samples)

    weighted_avg = weighted_sum / total_samples

    return weighted_avg


def save_scores(scores_list, n_samples_list, results_path):

    logging.info("Saving simulation results..")
    results = [{"score": score, "n_samples": n_samples} for score, n_samples in zip(scores_list, n_samples_list)]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f)


def initialize_model(model_config_path):
    """
    Initialize a model based on the provided configuration file.

    Parameters:
    - config_path(str): Path to the configuration file.

    Returns:
    - torch.nn.Module: The initialized model.
    """
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    model = eval(model_config["model_class"])(**model_config["init_args"])
    return model

def save_model_config(model, init_args, config_dir):
    """
    Save the configuration of a model to a JSON file.
    Parameters:
    - init_args(dict): dictionary of parameters used to initialize the model.
    - model(nn.Module): model to be saved.
    - config_dir: directory to save the configuration file.
    """
    model_config = {
        "model_class": model.__class__.__name__,
        "init_args": init_args
    }

    config_path = os.path.join(config_dir, "config_1.json")
    os.makedirs(config_dir, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(model_config, f)


def get_n_params(model):
    """
    Get the number of model parameters.
    Parameters:
    - model(nn.Module): model whose parameters are counted.
    Returns:
    - int: number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters())


def read_dict(file_path):
    """Reads a JSON file and returns the data."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def get_active_messages_metadata(local_models_metadata, attacked_client_id, keep_round_ids, rounds_frac, use_isolate=False):
    """Create a dictionary for running the Attribute Inference Attack using the isolated trajectory of each client.
    Parameters:
        local_models_metadata (dict): A dictionary containing the metadata of the local models.
        attacked_client_id (str): The ID of the attacked client.
        keep_round_ids (set): A set of round IDs to keep.
        rounds_frac (float): Fraction of rounds to keep.
        use_isolate (bool): Boolean to indicate if using isolated models or active ones.
    Returns:
        client_messages_metadata(dict): A dictionary containing the metadata of the client messages.
        """

    rounds_id = [int(round_id) for round_id in keep_round_ids]
    rounds_id.sort()
    if use_isolate is True:
        local_models_metadata = swap_dict_levels(local_models_metadata)
        if rounds_frac == 1:
            rounds_id.remove(max(rounds_id))
        client_messages_metadata = {
            'global': {f'{rounds_id[i]}': local_models_metadata[f"{attacked_client_id}"][f'{i}'] for i in range(len(rounds_id))},
            'local': {f'{rounds_id[i]}': local_models_metadata[f"{attacked_client_id}"][f'{i + 1}'] for i in range(len(rounds_id))}
        }
    else:
        if rounds_frac == 1:
            rounds_id.remove(max(rounds_id))
        client_messages_metadata = {
            'global': {f'{rounds_id[i]}': local_models_metadata['server'][f"{attacked_client_id}"][f'{i}'] for i in range(len(rounds_id))},
            'local': {f'{rounds_id[i]}': local_models_metadata[f"{attacked_client_id}"][f'{i + 1}'] for i in range(len(rounds_id))}
        }
    return client_messages_metadata

def get_gpu():
    """
    Get the name of the GPU device if available.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return gpu_name
    else:
        return "No GPU found or CUDA is not available."
    

def evaluate_trainer(trainer, dataloader):  
    """
    Evaluate a trainer on a dataloader
        Args:
            trainer (Trainer): Trainer object
            dataloader (): DataLoader object
        
        Returns:
            avg_loss (float): average loss of the model on the dataloader.
            metric (float): metric of the model on the dataloader."""
    
    evaluation_trainer = copy.deepcopy(trainer)
    if trainer.criterion.__class__.__name__ == "BCEWithLogitsLoss":
        evaluation_trainer.criterion = nn.BCEWithLogitsLoss(reduction='mean')
    elif trainer.criterion.__class__.__name__ == "CrossEntropyLoss":
        evaluation_trainer.criterion = nn.CrossEntropyLoss(reduction='mean')
    elif trainer.criterion.__class__.__name__ == "MSELoss":
        evaluation_trainer.criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Criterion {trainer.criterion.__class__.__name__} is not implemented.")
    avg_loss, metric = evaluation_trainer.evaluate_loader(dataloader)
    return avg_loss, metric


# TODO: test this function
def update_aia_results_dict(scores_dict, metrics_dict, loss_dict, n_samples_list, results_dict, iteration_id):
    """
    Update the results dictionary with the scores, metrics, and losses of the model-based AIA for a given iteration.
    Parameters:
    - scores_dict (dict): A dictionary mapping client IDs to the scores of the attack.
    - metrics_dict (dict): A dictionary mapping client IDs to the accuracy of each model.
    - loss_dict (dict): A dictionary mapping client IDs to the losses of each model.
    - n_samples_list (list): A list of the number of samples per client.
    - results_dict (dict): A dictionary containing the results of the attack.
    - iteration_id (int): The ID of the iteration.
    Returns:
    - results_dict (dict): A dictionary containing the results of the attack.
    """

    model_types = list(scores_dict.keys())

    for model_type in model_types:
        if model_type not in results_dict[iteration_id]:
            scores, metrics, losses = get_aia_scores(scores_dict, metrics_dict, loss_dict, model_type)
            results_dict[iteration_id][model_type] = {
                "scores": scores,
                "metrics": metrics,
                "losses": losses,
                "n_samples": n_samples_list
            }
    
    return results_dict


def get_aia_scores(scores_dict, metrics_dict, loss_dict, model_type):
    """
    Get the scores, metrics, and losses of the model-based AIA for a given model type.
    Parameters:
    - scores_dict (dict): A dictionary mapping client IDs to the scores of the attack.
    - metrics_dict (dict): A dictionary mapping client IDs to the accuracy of each model.
    - loss_dict (dict): A dictionary mapping client IDs to the losses of each model.
    - model_type (str): The type of the model. 
    Returns:
    - scores (list): A list of the scores of the attack.
    - metrics (list): A list of the metrics of the models.
    - losses (list): A list of the losses of the models.
    """
    scores = list(scores_dict[model_type].values())
    metrics = list(metrics_dict[model_type].values())
    losses = list(loss_dict[model_type].values())

    return scores, metrics, losses

def log_results(results_dict, iteration_id):
    """
    Log the results of the model-based AIA for a given iteration.
    Parameters:
    - results_dict (dict): A dictionary containing the results of the attack.
    - iteration_id (int): The ID of the iteration.
    """
    model_types = list(results_dict[iteration_id].keys())

    logging.info(f"Scores for round {iteration_id}")

    for model_type in model_types:
        scores = results_dict[iteration_id][model_type]["scores"]
        metrics = results_dict[iteration_id][model_type]["metrics"]
        losses = results_dict[iteration_id][model_type]["losses"]
        n_samples = results_dict[iteration_id][model_type]["n_samples"]

        weighted_avg_score = weighted_average(scores, n_samples)
        weighted_avg_metric = weighted_average(metrics, n_samples)
        weighted_avg_loss = weighted_average(losses, n_samples)

        logging.info(f"Average loss for {model_type} model: {weighted_avg_loss:.4f}")
        logging.info(f"Average accuracy for {model_type} model: {weighted_avg_metric:.4f}")
        logging.info(f"Average attack accuracy for {model_type} model: {weighted_avg_score:.4f}")
        