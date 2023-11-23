"""Source Inference Attack Simulation Script

"""
import argparse

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from fedklearn.models.linear import LinearLayer
from fedklearn.trainer.trainer import Trainer
from fedklearn.attacks.sia import SourceInferenceAttack

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
        help="Task name. Possible are 'adult'.",
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
        "--metadata_dir",
        type=str,
        help="Metadata directory",
        required=True
    )

    parser.add_argument(
        "--use_oracle",
        help='If chosen the optimal local mode will be used instead of the final updated model',
        action='store_true'
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
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


def load_models_metadata_dict(args):
    if args.use_oracle:
        with open(os.path.join(args.metadata_dir, "local.json"), "r") as f:
            models_metadata_dict = json.load(f)
    else:
        with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
            all_messages_metadata = json.load(f)

            last_round_id = max(map(int, all_messages_metadata["global"].keys()))

            all_messages_metadata.pop("global", None)

            models_metadata_dict = dict()

            for client_id in all_messages_metadata:
                models_metadata_dict[client_id] = all_messages_metadata[client_id][f"{last_round_id}"]

    return models_metadata_dict


def main():
    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)
    num_clients = len(federated_dataset.task_id_to_name)

    models_metadata_dict = load_models_metadata_dict(args)

    trainers_dict = initialize_trainers_dict(
        models_metadata_dict=models_metadata_dict, federated_dataset=federated_dataset, task_name=args.task_name,
        device=args.device
    )

    scores_list = []
    n_samples_list = []

    logging.info("Simulate Attacks..")

    for attacked_client_id in tqdm(range(num_clients)):

        logging.info("=" * 100)
        logging.info(f"Simulating attack for {attacked_client_id}...")

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=args.split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        attack_simulator = SourceInferenceAttack(
            attacked_client_id=attacked_client_id,
            dataloader=dataloader,
            trainers_dict=trainers_dict
        )

        attack_simulator.execute_attack()

        score = attack_simulator.evaluate_attack()

        logging.info(f"Score={score:.3f} for client {attacked_client_id}")

        scores_list.append(score)
        n_samples_list.append(len(dataset))

    logging.info("Save scores..")
    save_scores(scores_list=scores_list, n_samples_list=n_samples_list, results_path=args.results_path)
    logging.info(f"The results dictionary has been saved in {args.results_path}")


if __name__ == "__main__":
    main()
