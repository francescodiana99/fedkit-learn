"""Attribute Inference Attack Simulation Script

Implementation based on the technique present in (https://arxiv.org/abs/2108.06910)_
"""

import argparse

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.datasets.toy.toy import FederatedToyDataset
from fedklearn.models.linear import LinearLayer

from fedklearn.attacks.aia import AttributeInferenceAttack

from fedklearn.metrics import *


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
        "--task_name",
        type=str,
        choices=['adult', 'toy_regression', 'toy_classification'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification'.",
        required=True
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
        "--temperature",
        type=float,
        help="non-negative scalar temperature used for Gumbel-Softmax distribution.",
        default=1.0
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="non-negative scalar, between 0 and 1, used as a threshold in the binary case.",
        default=1.0
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Metadata file path",
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
        type=int, default=10,
        help="Logging frequency"
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="results.json",
        help="Path to the file where to save the results."
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


def initialize_dataset(args, rng):
    """
    Initialize the federated dataset based on the specified task.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        FederatedDataset: Initialized federated dataset.
    """
    if args.task_name == "adult":
        return FederatedAdultDataset(
            cache_dir=args.data_dir,
            download=False,
            rng=rng
        )
    elif args.task_name == "toy_regression" or args.task_name == "toy_classification":
        return FederatedToyDataset(
            cache_dir=args.data_dir,
            allow_generation=False
        )
    else:
        raise NotImplementedError(
            f"Dataset initialization for task '{args.task_name}' is not implemented."
        )


def main():
    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed)

    federated_dataset = initialize_dataset(args, rng)
    num_clients = len(federated_dataset.task_id_to_name)

    with open(args.metadata_path, "r") as f:
        all_messages_metadata = json.load(f)

    if args.task_name == "adult":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        model_init_fn = lambda: LinearLayer(input_dimension=41, output_dimension=1)
        is_binary_classification = True
        sensitive_attribute_type = args.sensitive_attribute_type
    elif args.task_name == "toy_classification":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        model_init_fn = lambda: LinearLayer(input_dimension=41, output_dimension=1)
        is_binary_classification = True
        sensitive_attribute_type = federated_dataset.sensitive_attribute_type
    elif args.task_name == "toy_regression":
        criterion = nn.MSELoss().to(args.device)
        model_init_fn = lambda: LinearLayer(input_dimension=41, output_dimension=1)
        is_binary_classification = False
        sensitive_attribute_type = federated_dataset.sensitive_attribute_type
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )

    success_metric = threshold_binary_accuracy if sensitive_attribute_type == "binary" else mean_squared_error

    scores_list = []
    n_samples_list = []

    logging.info("Simulate Attacks..")

    for attacked_client_id in tqdm(range(num_clients)):

        logging.info("=" * 100)
        logging.info(f"Simulating attack for {attacked_client_id}...")

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{attacked_client_id}"))

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=args.split)

        sensitive_attribute_id = dataset.column_name_to_id[args.sensitive_attribute]

        client_messages_metadata = {
            "global": all_messages_metadata["global"],
            "local": all_messages_metadata[f"{attacked_client_id}"]
        }

        attack_simulator = AttributeInferenceAttack(
            messages_metadata=client_messages_metadata,
            dataset=dataset,
            sensitive_attribute_id=sensitive_attribute_id,
            sensitive_attribute_type=sensitive_attribute_type,
            initialization=args.initialization,
            gumbel_temperature=args.temperature,
            gumbel_threshold=args.threshold,
            device=args.device,
            model_init_fn=model_init_fn,
            criterion=criterion,
            is_binary_classification=is_binary_classification,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            success_metric=success_metric,
            logger=logger,
            log_freq=args.log_freq,
            rng=rng,
            torch_rng=torch_rng
        )

        attack_simulator.execute_attack(num_iterations=args.num_rounds)

        score = attack_simulator.evaluate_attack()

        logging.info(f"Score={score:.3f} for client {attacked_client_id}")

        scores_list.append(score)
        n_samples_list.append(len(dataset))

    logging.info("Save scores..")
    save_scores(scores_list=scores_list, n_samples_list=n_samples_list, results_path=args.results_path)
    logging.info(f"The results dictionary has been saved in {args.results_path}")


if __name__ == "__main__":
    main()
