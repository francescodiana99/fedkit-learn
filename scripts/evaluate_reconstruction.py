import argparse

import numpy as np

from torch.utils.data import DataLoader

from fedklearn.utils import model_jsd

from utils import *
from constants import *

from tqdm import tqdm


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
        help="Task name. Possible are 'adult'.",
        required=True
    )
    parser.add_argument(
        "--models_metadata_path",
        type=str,
        help="Path to models metadata.",
        required=True
    )
    parser.add_argument(
        "--reference_models_metadata_path",
        type=str,
        help="Path to reference models metadata.",
        required=True
    )

    parser.add_argument(
        "--models_config_metadata_path",
        type=str,
        help="Path to the model config metadata.",
        required=True
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="Directory to cache data"
    )
    parser.add_argument(
        '--split',
        choices=['train', 'test'],
        default='train',
        help='Specify the split (train or test)'
    )

    parser.add_argument(
        "--sensitive_attribute",
        type=str,
        help="name of the sensitive attribute",
    )
    parser.add_argument(
        "--sensitive_attribute_type",
        type=str,
        help="type of the sensitive attribute. Possible are 'binary', 'categorical', and 'continuous'",
        choices=['binary', 'categorical', 'continuous'],
    )
    parser.add_argument(
        "--initialization",
        type=str,
        help="strategy used to initialize the sensitive attribute.",
        choices=['normal'],
        default="normal"
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
        "--num_rounds",
        type=int,
        default=100,
        help="Number of simulation rounds"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024, help="Batch size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the the directory where to save the results."
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


def compute_scores(
        task_name, federated_dataset, sensitive_attribute, sensitive_attribute_type, split, batch_size,
        reference_trainers_dict, trainers_dict, criterion, is_binary_classification, learning_rate, optimizer_name,
        aia_initialization, aia_num_rounds, device, rng, torch_rng
):
    task_type = get_task_type(task_name)

    scores_per_attack_dict = {f"{attack}": [] for attack in ATTACKS}
    n_samples_list = []

    logging.info("Simulate Attacks..")

    num_clients = len(trainers_dict)

    for attacked_client_id in tqdm(range(num_clients)):
        logging.info("=" * 100)
        logging.info(f"Simulating attacks for client {attacked_client_id}...")

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if task_name == "adult":
            # TODO: hard-code sensitive attribute_id and type in the federated_dataset
            sensitive_attribute_id = dataset.column_name_to_id[sensitive_attribute]
            sensitive_attribute_type = sensitive_attribute_type
        elif task_name == "toy_classification" or task_name == "toy_regression":
            sensitive_attribute_id = federated_dataset.sensitive_attribute_id
            sensitive_attribute_type = federated_dataset.sensitive_attribute_type
        else:
            raise NotImplementedError(
                f"Dataset initialization for task '{task_name}' is not implemented."
            )

        success_metric = threshold_binary_accuracy if sensitive_attribute_type == "binary" else mean_squared_error

        try:
            model = trainers_dict[f"{attacked_client_id}"].model
        except KeyError:
            model = trainers_dict[attacked_client_id].model

        try:
            reference_model = reference_trainers_dict[f"{attacked_client_id}"].model
        except KeyError:
            reference_model = reference_trainers_dict[attacked_client_id].model

        model.eval()
        reference_model.eval()

        sia_score = evaluate_sia(
            attacked_client_id=attacked_client_id, dataloader=dataloader, trainers_dict=trainers_dict
        )

        logging.info(f"SIA Score={sia_score:.3f} for client {attacked_client_id}")

        lmra_score = model_jsd(
            model, reference_model, dataloader=dataloader, task_type=task_type, device=device, epsilon=EPSILON
        )

        logging.info(f"LMRA Score={lmra_score:.3f} for client {attacked_client_id}")

        aia_score = evaluate_aia(
            model=model, dataset=dataset, sensitive_attribute_id=sensitive_attribute_id,
            sensitive_attribute_type=sensitive_attribute_type, initialization=aia_initialization, device=device,
            num_iterations=aia_num_rounds, criterion=criterion, is_binary_classification=is_binary_classification,
            learning_rate=learning_rate, optimizer_name=optimizer_name, success_metric=success_metric,
            rng=rng, torch_rng=torch_rng
        )

        aia_reference_score = evaluate_aia(model=reference_model, dataset=dataset,
            sensitive_attribute_id=sensitive_attribute_id, sensitive_attribute_type=sensitive_attribute_type,
            initialization=aia_initialization, device=device, num_iterations=aia_num_rounds, criterion=criterion,
            is_binary_classification=is_binary_classification, learning_rate=learning_rate,
            optimizer_name=optimizer_name, success_metric=success_metric, rng=rng, torch_rng=torch_rng)


        scores_per_attack_dict["SIA"].append(sia_score)
        scores_per_attack_dict["AIA"].append(aia_score)
        scores_per_attack_dict["LMRA"].append(lmra_score)

        logging.info(f"AIA Score={aia_score:.3f} for client {attacked_client_id}")
        logging.info(f"AIA Reference Score={aia_reference_score:.3f} for client {attacked_client_id}")

        n_samples_list.append(len(dataset))

    return scores_per_attack_dict, n_samples_list


def main():
    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)

    with open(args.models_metadata_path, "r") as f:
        models_metadata_dict = json.load(f)

    with open(args.reference_models_metadata_path, "r") as f:
        reference_models_metadata_dict = json.load(f)

    with open(args.models_config_metadata_path, "r") as f:
        model_config_metadata_dict = json.load(f)

    criterion, model_init_fn, is_binary_classification, metric = get_trainer_parameters(
        task_name=args.task_name, federated_dataset=federated_dataset, device=args.device,
        model_config_path=model_config_metadata_dict["model_config"]
    )

    reference_trainers_dict = initialize_trainers_dict(
        reference_models_metadata_dict, criterion=criterion, model_init_fn=model_init_fn,
        is_binary_classification=is_binary_classification, metric=metric, device=args.device
    )

    trainers_dict = initialize_trainers_dict(
        models_metadata_dict, criterion=criterion, model_init_fn=model_init_fn,
        is_binary_classification=is_binary_classification, metric=metric, device=args.device
    )

    scores_per_attack_dict, n_samples_list = compute_scores(
        task_name=args.task_name,
        federated_dataset=federated_dataset,
        sensitive_attribute=args.sensitive_attribute,
        sensitive_attribute_type=args.sensitive_attribute_type,
        split=args.split,
        batch_size=args.batch_size,
        reference_trainers_dict=reference_trainers_dict,
        trainers_dict=trainers_dict,
        criterion=criterion,
        is_binary_classification=is_binary_classification,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        aia_initialization=args.initialization,
        aia_num_rounds=args.num_rounds,
        device=args.device,
        rng=rng, torch_rng=torch_rng
    )

    logging.info("Saving scores..")
    os.makedirs(args.results_dir, exist_ok=True)

    for attack_name in scores_per_attack_dict:
        logging.info(f"=" * 100)
        logging.info(f"Save scores for {attack_name}")

        scores_list = scores_per_attack_dict[attack_name]
        results_path = os.path.join(args.results_dir, f"{attack_name}.json")
        save_scores(scores_list=scores_list, n_samples_list=n_samples_list, results_path=results_path)

        results_history_path = os.path.join(os.path.dirname(args.results_dir), "attacks_history.json")
        load_and_save_result_history(args.data_dir, scores_list, results_history_path, attack_name, n_samples_list)



if __name__ == "__main__":
    main()

