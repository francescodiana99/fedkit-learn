"""Local Model Reconstruction Attack Simulation Script

"""
import argparse

import numpy as np

from tqdm import tqdm

import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedklearn.attacks.lmra import LocalModelReconstructionAttack, GradientOracle

from fedklearn.models.sequential import SequentialNet

from utils import *
from constants import *


SCALING_COEFF = 10.


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
        '--hidden_layers',
        type=int,
        nargs='+',
        default=[],
        help='List representing the number of neurons in each hidden layer of the gradient prediction model'
    )

    parser.add_argument(
        "--use_oracle",
        action="store_true",
        help="If selected, use gradient oracle; otherwise, use gradient predictor."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer"
    )
    parser.add_argument(
        "--estimation_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate used to train the gradient estimator."
    )
    parser.add_argument(
        "--reconstruction_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate used to reconstruct the model; i.e., used to minimize the norm of the estimated gradient."
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="momentum"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.,
        help="Weight decay"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size"
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
        type=int,
        default=10,
        help="Logging frequency"
    )

    parser.add_argument(
        "--reconstructed_models_dir",
        type=str,
        default="./reconstructed_models",
        help="Directory to save reconstructed models."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Saving frequency."
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
        '--debug',
        help="Flag to use debug mode.",
        action="store_true"
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


def initialize_gradient_prediction_trainer(args, federated_dataset):
    if args.task_name == "adult":
        n_features = 41 + 1  # +1 because of the bias term
    elif args.task_name == "toy_classification":
        n_features = federated_dataset.n_features + 1  # +1 because of the bias term
    elif args.task_name == "toy_regression":
        n_features = federated_dataset.n_features + 1  # +1 because of the bias term
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )

    gradient_prediction_model = SequentialNet(
        input_dimension=n_features, output_dimension=n_features, hidden_layers=args.hidden_layers
    )

    gradient_prediction_model = gradient_prediction_model.to(args.device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [param for param in gradient_prediction_model.parameters() if param.requires_grad],
            lr=args.estimation_learning_rate,
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

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)
    num_clients = len(federated_dataset.task_id_to_name)

    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        all_messages_metadata = json.load(f)

    with open(os.path.join(args.metadata_dir, "local.json"), "r") as f:
        reference_models_metadata_dict = json.load(f)

    if args.task_name == "adult":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "toy_classification":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "toy_regression":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = False
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )
    model_init_fn = lambda: initialize_model(os.path.join(args.metadata_dir, "model_config.json"))
    logging.info(f"Loading model configuration from {os.path.join(args.metadata_dir, 'model_config.json')}")

    task_type = get_task_type(args.task_name)

    logging.info("Simulate Attacks..")

    os.makedirs(args.reconstructed_models_dir, exist_ok=True)

    # reconstructed_models_metadata_dict stores all the reconstruction trajectory.
    # final_reconstructed_models_dict only stores the final reconstructed models.
    all_reconstructed_models_metadata_dict = dict()
    final_reconstructed_models_metadata_dict = dict()

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

        gradient_prediction_trainer = initialize_gradient_prediction_trainer(args, federated_dataset=federated_dataset)

        if args.use_oracle or args.debug:
            gradient_oracle = GradientOracle(
                model_init_fn=model_init_fn, dataset=dataset, criterion=criterion,
                is_binary_classification=is_binary_classification, device=args.device
            )
        else:
            gradient_oracle = None

        attack_simulator = LocalModelReconstructionAttack(
            messages_metadata=client_messages_metadata,
            model_init_fn=model_init_fn,
            gradient_prediction_trainer=gradient_prediction_trainer,
            gradient_oracle=gradient_oracle,
            optimizer_name=args.optimizer,
            learning_rate=args.reconstruction_learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dataset=dataset,
            logger=logger,
            log_freq=args.log_freq,
            rng=rng
        )

        save_dir = os.path.join(args.reconstructed_models_dir, f"{attacked_client_id}")
        os.makedirs(save_dir, exist_ok=True)

        all_reconstructed_models_metadata_dict[f"{attacked_client_id}"] = attack_simulator.execute_attack(
            num_iterations=args.num_rounds, use_gradient_oracle=args.use_oracle,
            save_dir=save_dir, save_freq=args.save_freq,
            debug=args.debug, scaling_coeff=SCALING_COEFF  # TODO: read scaling coeff from metadata
        )

        last_saved_iteration = max(all_reconstructed_models_metadata_dict[f"{attacked_client_id}"], key=int)

        final_reconstructed_models_metadata_dict[f"{attacked_client_id}"] = (
            all_reconstructed_models_metadata_dict[f"{attacked_client_id}"][last_saved_iteration]
        )

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

    # Swaps the levels of keys in all_reconstructed_models_metadata_dict.
    # In the new dictionary, the first level represents iteration ids and the second level represents client ids.
    all_reconstructed_models_metadata_dict = swap_dict_levels(all_reconstructed_models_metadata_dict)

    # Saving results
    save_scores(scores_list=scores_list, n_samples_list=n_samples_list, results_path=args.results_path)

    logging.info("=" * 100)
    logging.info("Save trajectory of all reconstructed models metadata..")

    all_reconstructed_models_metadata_path = os.path.join(args.metadata_dir, "trajectory.json")
    with open(all_reconstructed_models_metadata_path, "w") as f:
        json.dump(all_reconstructed_models_metadata_dict, f)

    logging.info(
        f"The trajectories of all reconstructed models metadata is saved to {all_reconstructed_models_metadata_path}"
    )

    logging.info("=" * 100)
    logging.info("Save final reconstructed models metadata..")

    final_reconstructed_models_metadata_path = os.path.join(args.metadata_dir, "reconstructed.json")
    with open(final_reconstructed_models_metadata_path, "w") as f:
        json.dump(final_reconstructed_models_metadata_dict, f)

    logging.info(f"Final reconstructed models metadata is saved to {final_reconstructed_models_metadata_path}")


if __name__ == "__main__":
    main()
