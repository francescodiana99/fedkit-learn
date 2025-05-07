"""Model Based Attribute Inference Attack Evaluation Script.
This script evaluates the performance of the model-based attribute inference attack on different models.
"""

import argparse
import copy
import logging

import numpy as np

from torch.utils.data import DataLoader

from utils import *

from tqdm import tqdm

def parse_args(args_list=None):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="Path to the metadata file of the models",
        required=True
    )

    parser.add_argument(
        "--active_rounds",
        type=str,
        nargs='+',
        help="Active rounds to evaluate",
        default=None
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for training and testing",
        default="cpu"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the random number generator",
        default=42
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory to save the results",
        required=True
    )

    parser.add_argument(
        "--sensitive_attribute",
        type=str,
        help="name of the sensitive attribute",
        required=True
    )

    parser.add_argument(
        "--sensitive_attribute_type",
        type=str,
        choices=['binary', 'categorical', 'numerical'],
        help="type of the sensitive attribute. Possible are 'binary', 'categorical', and 'numerical'",
        required=True
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'test'],
        help="Specify the split (train or test)",
        default='train'
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=1e-2
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer",
        default="sgd"
    )

    parser.add_argument(
        "--initialization",
        type=str,
        help="strategy used to initialize the sensitive attribute.",
        choices=['normal'],
        default="normal"
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        help="Number of optimization rounds in case of numerical sensitive attribute",
        default=100
    )

    parser.add_argument(
        "--use_oracle",
        action="store_true",
        help="Flag to evauluate an oracle for the attack performance",
        default=False
    )

    parser.add_argument(
        '--attacked_task',
        type=int,
        help="Task to attack. If set, only this client will be evaluated",
        default=None,
    )

    parser.add_argument(
        "--attacked_round",
        type=str,
        help="Attacked rounds to test. In case of active rounds, it is the first round of the active attack.",
        required=True
    )

    parser.add_argument(
        "--use_isolated",
        action="store_true",
        help="Test models obtained by isolating the clients",
        default=False
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

def init_dict_from_chkpt(models_metadata, rounds):
    """Retrieve the metadata of the models from a specific checkpoint"""

    models_metadata_dict = dict()

    if "global" in models_metadata.keys():
        models_metadata.pop("global", None)
    if "server" in models_metadata.keys():
        models_metadata.pop("server", None)

    models_metadata = swap_dict_levels(models_metadata)
    models_metadata_dict = {k: models_metadata.get(k) for k in rounds}

    return models_metadata_dict

def get_scores(args, trainer, dataloader, sensitive_attribute_id,
              sensitive_attribute_type, criterion, cast_float, success_metric, 
              rng, torch_rng
              ):
    """Helper function to compute the reconstruction scores for a single model"""
    loss, metric = trainer.evaluate_loader(dataloader)
    aia_score = evaluate_mb_aia(model=trainer.model, 
                                dataset=dataloader.dataset,
                                sensitive_attribute_id=sensitive_attribute_id,
                                sensitive_attribute_type=sensitive_attribute_type,
                                initialization=args.initialization, 
                                device=args.device, 
                                num_iterations=args.num_rounds,
                                criterion=criterion,
                                cast_float=cast_float, 
                                learning_rate=args.learning_rate,
                                optimizer_name=args.optimizer, 
                                success_metric=success_metric, rng=rng,
                                torch_rng=torch_rng
                                )
    return loss, metric.item(), aia_score
    

def compute_scores(args, fl_setup, split, trainers_dict, criterion, cast_float, rng, torch_rng, 
                   oracle_trainers_dict=None, active_trainers_dict=None):
    """ Compute the reconstruction scores for each model
    Args:
        args (argparse.Namespace): parsed arguments
        fl_setup (dict): dictionary containing the federated learning setup
        split (str): split to use for evaluation. Possible values are 'train' and 'test'
        trainers_dict (dict): dictionary containing the trainer objects
        oracle_trainers_dict (dict): dictionary containing the trainer objects to use as a oracle
        active_trainers_dict (dict): dictionary containing the trainer objects of the active attack
        criterion (torch.nn.Module): loss function
        cast_float (bool): whether the target value should be cast to float
        rng (np.random.Generator): random number generator for reproducibility
        torch_rng (torch.Generator): torch random number generator for reproducibility
    
    Returns:
        scores_per_client_dict (dict): dictionary containing the scores for each client
        metrics_dict (dict): dictionary containing the metrics for each client
        loss_dict (dict): dictionary containing the losses for each client
        n_samples_list (list): list containing the number of samples for each client
    """

    logging.info(f"Simulate model-based AIA")
    
    n_samples_list = []
    num_clients = len(trainers_dict) - 1

    model_types = ["global", "client"]
    if args.use_oracle:
        model_types.append("oracle")
    if args.active_rounds is not None:
        model_types.append("active")
    scores_per_client_dict = {cat: {} for cat in model_types}
    metrics_dict = {cat: {} for cat in model_types}
    loss_dict = {cat: {} for cat in model_types}

    federated_dataset = load_dataset(fl_setup, rng=rng)

    global_trainer = trainers_dict["global"]

    pbar = tqdm(range(num_clients))
    attacked_task_id = 0
    if args.attacked_task is not None:
        attacked_task_id = args.attacked_task
        compute_single_client = True
    else:
        compute_single_client = False

    while attacked_task_id < num_clients:

        dataset = federated_dataset.get_task_dataset(attacked_task_id, mode=split)
        if fl_setup["task_name"] in ["adult", "purchase", "purchase_binary", "income", "medical_cost", "binary_income"]:
            sensitive_attribute_id = dataset.column_name_to_id[args.sensitive_attribute]
            sensitive_attribute_type = args.sensitive_attribute_type
        elif fl_setup["task_name"] in ["toy_classification", "toy_regression"]:
            sensitive_attribute_id = federated_dataset.sensitive_attribute_id
            sensitive_attribute_type = federated_dataset.sensitive_attribute_type
        else:
            raise NotImplementedError(
                f"Dataset initialization for task '{fl_setup["task_name"]}' is not implemented."
            )
        success_metric = threshold_binary_accuracy if sensitive_attribute_type == "binary" else mean_squared_error
        dataloader = DataLoader(dataset, batch_size=fl_setup["batch_size"])

        logging.info('=' * 50)

        models_to_evaluate = {
            "global": global_trainer,
            "client": trainers_dict[f"{attacked_task_id}"],
            "oracle": oracle_trainers_dict.get(f"{attacked_task_id}") if oracle_trainers_dict else None,
            "active": active_trainers_dict.get(f"{attacked_task_id}") if active_trainers_dict else None
        }

        for model_type, trainer in models_to_evaluate.items():
            if trainer:
                loss, metric, score = get_scores(
                    args=args,
                    trainer=trainer, 
                    dataloader=dataloader, 
                    sensitive_attribute_id=sensitive_attribute_id, 
                    sensitive_attribute_type=sensitive_attribute_type, 
                    criterion=criterion, 
                    cast_float=cast_float, 
                    success_metric=success_metric, 
                    rng=rng, 
                    torch_rng=torch_rng
                )
                loss_dict[model_type][attacked_task_id] = loss
                metrics_dict[model_type][attacked_task_id] = metric
                scores_per_client_dict[model_type][attacked_task_id] = score
                logging.info(f"Score={score:.4f} for task {attacked_task_id} with {model_type} model")  

        n_samples_list.append(len(dataset))
        attacked_task_id += 1
        pbar.update(1)
        if compute_single_client:
            attacked_task_id = num_clients
    pbar.close()
    return scores_per_client_dict, metrics_dict, loss_dict, n_samples_list

def update_aia_results_dict(scores_dict, metrics_dict, loss_dict, n_samples_list, results_dict, active_round=None):
    """
    Update the results dictionary with the scores, metrics, and losses of the model-based AIA for a given iteration.
    Parameters:
    - scores_dict (dict): A dictionary mapping client IDs to the reconstruction accuracy of the attack.
    - metrics_dict (dict): A dictionary mapping client IDs to the accuracy of each model.
    - loss_dict (dict): A dictionary mapping client IDs to the losses of each model.
    - n_samples_list (list): A list of the number of samples per client.
    - results_dict (dict): A dictionary containing the results of the attack.
    - active_round (int): The active round of the attack. If None, the attack is passive.
        
    Returns:
    - results_dict (dict): A dictionary containing the results of the attack.
    """

    results_dict["passive"]["global"] = {
        "loss": loss_dict["global"],
        "metric": metrics_dict["global"],
        "score": scores_dict["global"],
        "n_samples": n_samples_list
        }
    results_dict["passive"]["client"] = {
        "loss": loss_dict["client"],
        "metric": metrics_dict["client"],
        "score": scores_dict["client"],
        "n_samples": n_samples_list
        
    }

    if "oracle" in scores_dict.keys():
        results_dict["oracle"] = {
            "loss": loss_dict["oracle"],
            "metric": metrics_dict["oracle"],
            "score": scores_dict["oracle"],
            "n_samples": n_samples_list
        }
    if "active" in scores_dict.keys():
        results_dict["active"][f"{active_round}"] = {
            "loss": loss_dict["active"],
            "metric": metrics_dict["active"],
            "score": scores_dict["active"],
            "n_samples": n_samples_list
        }
    
    return results_dict

def log_model_results(model_data, model_type, active_round=None):
    """
    Helper function to log the average results for a given model type.

    Parameters:
    - model_data (dict): The dictionary containing scores, metrics, and losses.
    - model_type (str): The type of model being logged (e.g., "passive", "oracle", "active").
    """

    
    scores = list(model_data["score"].values())
    metrics = list(model_data["metric"].values())
    losses = list(model_data["loss"].values())
    n_samples = list(model_data["n_samples"])


    weighted_avg_score = weighted_average(scores, n_samples)
    weighted_avg_metric = weighted_average(metrics, n_samples)
    weighted_avg_loss = weighted_average(losses, n_samples)

    logging.info(f"Average loss for {model_type} model: {weighted_avg_loss:.4f}")
    logging.info(f"Average accuracy for {model_type} model: {weighted_avg_metric:.4f}")
    logging.info(f"Average attack accuracy for {model_type} model: {weighted_avg_score:.4f}")
    logging.info("+" * 50)


def log_aia_results(results_dict, active_round=None):
    """
    Log the results of the model-based AIA for a given iteration.
    Parameters:
    - results_dict (dict): A dictionary containing the results of the attack.
    - active_round (int): The active round of the attack. If None, the attack is passive.
    """
    if active_round is not None:
        logging.info(f"Scores for AIA attacks after {active_round} active rounds ")
    else:
        logging.info("Scores for passive AIA attacks")

    attack_types = [k for k in results_dict.keys() if len(results_dict[k]) != 0]

    for model_type in attack_types:
        if model_type == "active":
            log_model_results(results_dict["active"][f"{active_round}"], model_type=model_type)
        elif model_type == "passive":
            log_model_results(results_dict[model_type]["global"], model_type=model_type)
            log_model_results(results_dict[model_type]["client"], model_type=model_type)
        else:
            log_model_results(results_dict[model_type], model_type=model_type)

def main():

    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed)

    try:
        with open(os.path.join(args.metadata_dir, "setup.json"), "r") as f:
            fl_setup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Federated Learning simulation metadata file not found at \
                                '{args.metadata_dir}/setup.json'.")
    all_scores = {
        "passive": {},
        "oracle": {},
        "active": {}
    }

    metadata_path = os.path.join(args.metadata_dir, "federated.json")
    with open(metadata_path, "r") as f:
        all_messages_metadata = json.load(f)
        all_messages_metadata = swap_dict_levels(all_messages_metadata)
    passive_rounds_dict = all_messages_metadata[args.attacked_round]

    if args.active_rounds is not None:
        if args.use_isolated is True:
            with open(os.path.join(args.metadata_dir, f"isolated_trajectories_{args.attacked_round}.json"), "r") as f:
                all_active_rounds_metadata = json.load(f)
                active_rounds_dict = {k: all_active_rounds_metadata.get(k) for k in args.active_rounds}
        else:
            with open(os.path.join(args.metadata_dir, f"active_trajectories_{args.attacked_round}.json"), "r") as f:
                all_active_rounds_metadata = json.load(f)
                active_rounds_dict = init_dict_from_chkpt(all_active_rounds_metadata, args.active_rounds)

        
    criterion, metric, cast_float = get_trainers_config(fl_setup["task_name"])
    criterion = criterion.to(args.device)
    model_init_fn = lambda: initialize_model(fl_setup["model_config_path"])

    passive_trainers_dict = initialize_trainers_dict(
        passive_rounds_dict,
        criterion=criterion, 
        model_init_fn=model_init_fn,
        cast_float=cast_float, 
        metric=metric, 
        device=args.device
        )

    if args.use_oracle:
        with open(os.path.join(args.metadata_dir, "last_local.json"), "r") as f:
            oracle_models_metadata = json.load(f)
            oracle_trainers_dict = initialize_trainers_dict(
            oracle_models_metadata,
            criterion=criterion, 
            model_init_fn=model_init_fn,
            cast_float=cast_float, 
            metric=metric, 
            device=args.device
            )
    
    if args.active_rounds is None:
        scores_dict, metrics_dict, loss_dict, n_samples_list = compute_scores(
            fl_setup=fl_setup,
            args=args,
            split=args.split,
            trainers_dict=passive_trainers_dict,
            criterion=criterion,
            cast_float=cast_float,
            rng=rng,
            torch_rng=torch_rng,
            oracle_trainers_dict=oracle_trainers_dict if args.use_oracle else None,
            active_trainers_dict=None
            )
        all_scores = update_aia_results_dict(
            scores_dict=scores_dict,
            metrics_dict=metrics_dict,
            loss_dict=loss_dict,
            n_samples_list=n_samples_list,
            results_dict=all_scores,
            active_round=None
        )
        log_aia_results(all_scores)

    
    else:
        for active_round in args.active_rounds:
            active_trainers_dict = initialize_trainers_dict(
                active_rounds_dict[active_round],
                criterion=criterion, 
                model_init_fn=model_init_fn,
                cast_float=cast_float, 
                metric=metric, 
                device=args.device
                )
            scores_dict, metrics_dict, loss_dict, n_samples_list = compute_scores(
                fl_setup=fl_setup,
                args=args,
                split=args.split,
                trainers_dict=passive_trainers_dict,
                criterion=criterion,
                cast_float=cast_float,
                rng=rng,
                torch_rng=torch_rng,
                oracle_trainers_dict=oracle_trainers_dict if args.use_oracle else None,
                active_trainers_dict=active_trainers_dict
                )
            all_scores = update_aia_results_dict(
                scores_dict=scores_dict,
                metrics_dict=metrics_dict,
                loss_dict=loss_dict,
                n_samples_list=n_samples_list,
                results_dict=all_scores,
                active_round=active_round
            )
            log_aia_results(all_scores, active_round)    

    logging.info("Saving scores..")
    if args.use_isolated:
        results_path = os.path.join(args.results_dir, f"isolated_aia_mb_round_{args.attacked_round}.json")
    else:
        results_path = os.path.join(args.results_dir, f"aia_mb_round_{args.attacked_round}.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            scores_dict = json.load(f)
        scores_dict.update(all_scores)
    else:
        scores_dict = all_scores

    with open(results_path, "w") as f:
        json.dump(scores_dict, f)

    logging.info(f"Scores saved in {results_path}")

if __name__ == "__main__":
    main()
