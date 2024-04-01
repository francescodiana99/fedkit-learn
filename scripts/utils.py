import os
import json
import logging
from collections import defaultdict

from tqdm import tqdm

from fedklearn.datasets.purchase.purchase import FederatedPurchaseDataset
from fedklearn.metrics import *

from fedklearn.models.linear import LinearLayer, TwoLinearLayers
from fedklearn.models.sequential import SequentialNet
from fedklearn.trainer.trainer import Trainer, DebugTrainer

from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.datasets.toy.toy import FederatedToyDataset

from fedklearn.attacks.aia import ModelDrivenAttributeInferenceAttack
from fedklearn.attacks.sia import SourceInferenceAttack


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
    logging.basicConfig(level=logging.INFO - (args.verbose - args.quiet) * 10)

def get_task_type(task_name):
    if task_name == "adult":
        task_type = "binary_classification"
    elif task_name == "toy_classification":
        task_type = "binary_classification"
    elif task_name == "toy_regression":
        task_type = "regression"
    elif task_name == "purchase":
        task_type = "classification"
    else:
        raise NotImplementedError(
            f"Network initialization for task '{task_name}' is not implemented"
        )

    return task_type

def load_dataset(task_name, data_dir, rng):
    """
    Load a federated dataset based on the specified task name.

    Args:
        task_name (str): Name of the task for which the dataset is to be loaded.
        data_dir (str): Directory where the dataset should be stored or loaded from.
        rng (RandomState): NumPy random number generator for reproducibility.
        split_criterion (str): The split criterion to be used for the task data separation.

    Returns:
        FederatedDataset: Initialized federated dataset.

    Raises:
        NotImplementedError: If the dataset initialization for the specified task is not implemented.
    """
    if task_name == "adult":
        with open(os.path.join(data_dir, "split_criterion.json"), "r") as f:
            split_dict = json.load(f)
        split_criterion = split_dict["split_criterion"]

        if split_criterion is None:
            raise ValueError("Split criterion must be specified for the Adult dataset.")
        if split_criterion == "n_tasks":
            n_tasks = split_dict["n_tasks"]
            n_task_samples = split_dict["n_task_samples"]
            return FederatedAdultDataset(
                cache_dir=data_dir,
                download=False,
                rng=rng,
                split_criterion=split_criterion,
                n_tasks=n_tasks,
                n_task_samples=n_task_samples
            )
        return FederatedAdultDataset(
            cache_dir=data_dir,
            download=False,
            rng=rng,
            split_criterion=split_criterion,

        )
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

    elif task_name == "toy_regression" or task_name == "toy_classification":
        return FederatedToyDataset(
            cache_dir=data_dir,
            allow_generation=False,
            force_generation=False,
            rng=rng
        )
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
        model_chkpts = torch.load(models_metadata_dict[client_id])["model_state_dict"]
        model = model_init_fn()
        model.load_state_dict(model_chkpts)

        trainers_dict[client_id] = DebugTrainer(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            is_binary_classification=is_binary_classification
        )

    return trainers_dict


def evaluate_sia(attacked_client_id, dataloader, trainers_dict):
    """
    Evaluate Source Inference Attack.

    Parameters:
    - attacked_client_id (str): The ID of the attacked client.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
    - trainers_dict (Dict[str, Trainer]): A dictionary mapping model IDs to Trainer objects.

    Returns:
    - float: The evaluation score for the Source Inference Attack.
    """
    attack_simulator = SourceInferenceAttack(
        attacked_client_id=attacked_client_id,
        dataloader=dataloader,
        trainers_dict=trainers_dict
    )

    attack_simulator.execute_attack()

    score = attack_simulator.evaluate_attack()

    return score


def evaluate_aia(
        model, dataset, sensitive_attribute_id, sensitive_attribute_type, initialization, device, num_iterations,
        criterion, is_binary_classification, learning_rate, optimizer_name, success_metric, rng=None, torch_rng=None,
        output_losses=False, output_predictions=False
):

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

    if output_losses:
        logging.info(f"{all_losses[:20]}")

    if output_predictions:
        logging.info(f"{all_losses[:20].argmax(axis=1)}")
    return float(score)

def split_data_by_aia(reference_trainers_dict, global_trainer, federated_dataset, sensitive_attribute,
                      sensitive_attribute_type, initialization, device,criterion,
                      is_binary_classification, learning_rate, optimizer_name, success_metric, data_path,
                      rng=None, torch_rng=None, verbose=False):
    """
    Split data in multiple datasets based on the AIA results.
    """
    global_model = global_trainer.model
    global_model.eval()
    # df_dict = get_aia_split(model=global_model, dataset=dataset, sensitive_attribute_id=sensitive_attribute_id,
    #                         sensitive_attribute_type=sensitive_attribute_type, initialization=initialization,
    #                         device=device, criterion=criterion, is_binary_classification=is_binary_classification,
    #                         learning_rate=learning_rate, optimizer_name=optimizer_name, success_metric=success_metric,
    #                         rng=rng, torch_rng=torch_rng, verbose=verbose)

    # for key in df_dict.keys():
    #     df_dict[key].to_csv(f"{data_path}/global/{key}.csv", index=False)

    num_clients = len(reference_trainers_dict)

    for attacked_client_id in tqdm(range(num_clients)):
        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode="train")
        sensitive_attribute_id = dataset.column_name_to_id[sensitive_attribute]
        attacked_model = reference_trainers_dict[f"{attacked_client_id}"].model
        attacked_model.eval()
        df_dict_global = get_aia_split(model=global_model, dataset=dataset, sensitive_attribute_id=sensitive_attribute_id,
                                sensitive_attribute_type=sensitive_attribute_type, initialization=initialization,
                                device=device, criterion=criterion, is_binary_classification=is_binary_classification,
                                learning_rate=learning_rate, optimizer_name=optimizer_name,
                                success_metric=success_metric,
                                rng=rng, torch_rng=torch_rng, verbose=verbose)
        df_dict = get_aia_split(model=attacked_model, dataset=dataset,
                                sensitive_attribute_id=sensitive_attribute_id,
                                sensitive_attribute_type=sensitive_attribute_type, initialization=initialization,
                                device=device, criterion=criterion, is_binary_classification=is_binary_classification,
                                learning_rate=learning_rate, optimizer_name=optimizer_name, success_metric=success_metric,
                                rng=rng, torch_rng=torch_rng, verbose=verbose)
        for key in df_dict.keys():
            os.makedirs(os.path.join(data_path, f"{attacked_client_id}"), exist_ok=True)
            df_dict[key].to_csv(f"{data_path}/{attacked_client_id}/{key}.csv", index=False)

            # TODO: fix this
        for key in df_dict_global.keys():
            os.makedirs(os.path.join(data_path, "global", f"{attacked_client_id}",), exist_ok=True)
            df_dict_global[key].to_csv(f"{data_path}/global/{attacked_client_id}/{key}.csv", index=False)


def get_aia_split(model, dataset, sensitive_attribute_id, sensitive_attribute_type, initialization, device,
                  criterion, is_binary_classification, learning_rate, optimizer_name,
                  success_metric, rng=None, torch_rng=None, verbose=False
                  ):

    """
    Split data according to the AIA.
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

    return attack_simulator.execute_attack_and_split_data(verbose=verbose)

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


def save_avg_scores_adult(scores_list, attack_name, results_path, n_samples_list, n_tasks, split_criterion, seed):

    seed = str(seed)
    avg_score = weighted_average(scores=scores_list, n_samples=n_samples_list)

    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = defaultdict()
    if seed not in results:
        results[seed] = dict()

    if split_criterion == ["n_task_samples"]:
        results[seed][split_criterion][str(n_tasks)][str(n_samples_list[-1])][attack_name] = avg_score
    else:
        results[seed][split_criterion][str(n_tasks)][attack_name] = avg_score
    with open(results_path, 'w') as f:
        json.dump(results, f)

    logging.info(f"Average Score={avg_score:.3f}")


def load_and_save_result_history(data_dir, scores_list, results_path, attack_name, n_samples_list,seed ):
    """Save average results for all the attacks in a json file."""

    with open(os.path.join(data_dir, "split_criterion.json"), "r") as f:
        split_dict = json.load(f)
    split_criterion = split_dict["split_criterion"]
    n_tasks = split_dict["n_tasks"]
    save_avg_scores_adult(scores_list=scores_list, attack_name=attack_name, results_path=results_path,
                          n_samples_list=n_samples_list, n_tasks=n_tasks, split_criterion=split_criterion, seed=seed)



    logging.info(f"The results dictionary has been saved in {results_path}")

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

