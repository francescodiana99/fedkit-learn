"""Local Model Reconstruction Attack Simulation Script

"""
import argparse

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.models.linear import LinearLayer
from fedklearn.trainer.trainer import Trainer

from fedklearn.attacks.lmra import LocalModelReconstructionAttack

from fedklearn.metrics import *


from utils import *
from constants import *


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
        "--metadata_dir",
        type=str,
        help="Metadata directory",
        required=True
    )

    parser.add_argument(
        "--download_data",
        help='If chosen the dataset is downloaded if not found.',
        action='store_true'
    )
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
        "--momentum",
        type=float,
        default=0.9,
        help="momentum"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024, help="Batch size"
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=200,
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
        "--reconstructed_models_dir",
        type=str,
        default="./reconstructed_models",
        help="Directory to save reconstructed models."
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
            test_frac=args.test_frac,
            drop_nationality=not args.use_nationality,
            scaler_name=args.scaler_name,
            rng=rng
        )
    else:
        raise NotImplementedError(
            f"Dataset initialization for task '{args.task_name}' is not implemented."
        )


def initialize_gradient_prediction_trainer(args):
    if args.task_name == "adult":
        gradient_prediction_model = LinearLayer(input_dimension=42, output_dimension=42).to(args.device)
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [param for param in gradient_prediction_model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{args.optimizer}' is not implemented"
        )

    return Trainer(
        model=gradient_prediction_model,
        criterion=nn.MSELoss().to(args.device),
        metric=mean_squared_error,
        device=args.device,
        optimizer=optimizer,
        is_binary_classification=False
    )


def main():
    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)

    federated_dataset = initialize_dataset(args, rng)
    num_clients = len(federated_dataset.task_id_to_name)

    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        all_messages_metadata = json.load(f)

    with open(os.path.join(args.metadata_dir, "local.json"), "r") as f:
        reference_models_metadata_dict = json.load(f)

    if args.task_name == "adult":
        model_init_fn = lambda: LinearLayer(input_dimension=41, output_dimension=1)
        task_type = "binary_classification"
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )

    logging.info("Simulate Attacks..")

    os.makedirs(args.reconstructed_models_dir, exist_ok=True)

    reconstructed_models_metadata_dict = dict()
    scores_list = []
    n_samples_list = []

    for attacked_client_id in tqdm(range(num_clients)):
        logging.info("=" * 100)
        logging.info(f"Simulating attack for {attacked_client_id}...")

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{attacked_client_id}"))

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=args.split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        client_messages_metadata = {
            "global": all_messages_metadata["global"],
            "local": all_messages_metadata[f"{attacked_client_id}"]
        }

        gradient_prediction_trainer = initialize_gradient_prediction_trainer(args)

        attack_simulator = LocalModelReconstructionAttack(
            messages_metadata=client_messages_metadata,
            model_init_fn=model_init_fn,
            gradient_prediction_trainer=gradient_prediction_trainer,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dataset=dataset,
            logger=logger,
            log_freq=args.log_freq,
            rng=rng
        )

        reconstructed_model = attack_simulator.execute_attack(num_iterations=args.num_rounds)
        logging.info("Local model reconstructed successfully.")

        logging.info("=" * 50)
        logging.info("Load reference model..")
        reference_model = attack_simulator.model_init_fn()
        model_chkpts = torch.load(reference_models_metadata_dict[f"{attacked_client_id}"])["model_state_dict"]
        reference_model.load_state_dict(model_chkpts)
        logging.info("Reference model loaded successfully.")

        logging.info("=" * 50)
        logging.info("Evaluating the reconstructed model..")
        score = attack_simulator.evaluate_attack(
            reference_model=reference_model,
            dataloader=dataloader,
            task_type=task_type,
            epsilon=EPSILON
        )

        logging.info(f"Score={score:.3f} for client {attacked_client_id}")

        scores_list.append(score)
        n_samples_list.append(len(dataset))

        logging.info("=" * 50)
        logging.info(f"Save reconstructed model for client {attacked_client_id}..")
        checkpoint = {'model_state_dict': reconstructed_model.state_dict()}

        path = os.path.join(args.reconstructed_models_dir, f"{attacked_client_id}.pt")
        path = os.path.abspath(path)
        torch.save(checkpoint, path)

        reconstructed_models_metadata_dict[f"{attacked_client_id}"] = path

    save_scores(scores_list=scores_list, n_samples_list=n_samples_list, results_path=args.results_path)
    
    logging.info("=" * 100)
    logging.info("Save reconstructed models metadata..")
    reconstructed_models_metadata_path = os.path.join(args.metadata_dir, "reconstructed.json")
    with open(reconstructed_models_metadata_path, "w") as f:
        json.dump(reconstructed_models_metadata_dict, f)
    logging.info(f"Reconstructed models metadata is save to {reconstructed_models_metadata_path}")


