import argparse
import os
import logging

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from fedklearn.metrics import multiclass_accuracy


from tqdm import tqdm

from fedklearn.utils import get_param_tensor
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
    '--data_dir',
            type=str, default='./data/',
            help='Directory containing the dataset.'
            )


    parser.add_argument(
        '--task_name',
        type=str,
        choices=['adult', 'purchase', 'toy_classification', 'toy_regression', 'purchase_binary', 'medical_cost',
                 'income'],
        help="Task name. Possible choices are 'adult', 'purchase', 'toy_classification', "
             "'toy_regression', 'purchase_binary', 'medical_cost', 'income'",
        required=True)

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
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer"
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
        "--num_epochs",
        type=int,
        default=20,
        help="Number of finetuning epochs"
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
        "--finetuned_models_dir",
        type=str,
        default="./finetuned_models",
        help="Directory to save finetuned models."
    )

    parser.add_argument(
        "--finetune_round",
        required=True,
        type=int,
        help="Starting round for finetuning."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="Saving frequency."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--compute_single_client",
        action="store_true",
        help="Compute a single client"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument(
    "--quiet",
    action="store_true",
    )

    parser.add_argument(
        "--noise_factor",
        type=float,
        default=None,
        help="Noise factor to simulate server update")

    parser.add_argument('--mixing_coefficient',
                        type=float,
                        default=None,
                        help='Mixing coefficient for the correlation-based split in Adult dataset')

    parser.add_argument(
        '--state',
        type=str,
        default='None',
        help='USA state to select Income dataset subset'
    )


    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def initialize_finetuning_trainer(args, client_messages_metadata, model_init_fn, criterion, metric,
                                  is_binary_classification):
    """
    Initialize the trainer for finetuning.

    """
    global_model_chkpt = torch.load(client_messages_metadata['global'][f"{args.finetune_round}"],
                                    map_location=args.device)["model_state_dict"]
    finetuning_model = model_init_fn()
    finetuning_model.load_state_dict(global_model_chkpt)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [param for param in finetuning_model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            [param for param in finetuning_model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{args.optimizer}' is not implemented"
        )

    return Trainer(
        model=finetuning_model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        is_binary_classification=is_binary_classification,
        metric=metric

    )

def add_noise(model, noise_model, noise_factor):
    """
    Add noise to the model weights.

    """
    for layer1, layer2 in zip(model.layers, noise_model.layers):
        if isinstance(layer1, torch.nn.Linear) and isinstance(layer2, torch.nn.Linear):
            assert layer1.weight.shape == layer2.weight.shape, "Model layers must have the same shape"
            layer1.weight.data += (layer2.weight.data * noise_factor)
            layer1.bias.data += (layer2.bias.data * noise_factor)

    return model

def main():
    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng,
                                     mixing_coefficient=args.mixing_coefficient, state=args.state)

    num_clients = len(federated_dataset.task_id_to_name)

    with open(os.path.join(args.metadata_dir, "federated.json"), "r") as f:
        all_messages_metadata = json.load(f)

    # TODO : fix is_binary_classification for regression tasks
    if args.task_name == "adult":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "toy_classification":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "toy_regression":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "purchase":
        criterion = nn.CrossEntropyLoss().to(args.device)
        is_binary_classification = False
    elif args.task_name == "purchase_binary":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "medical_cost":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "income":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = True
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )
    with open(os.path.join(args.metadata_dir, "model_config.json"), "r") as f:
        model_config_path = json.load(f)
    model_init_fn = lambda: initialize_model(model_config_path["model_config"])

    logging.info("Simulate Attacks..")

    os.makedirs(args.finetuned_models_dir, exist_ok=True)

    all_finetuned_models_metadata_dict = defaultdict(lambda : dict())
    final_finetuned_models_metadata_dict = defaultdict(lambda : dict())

    pbar = tqdm(range(num_clients))
    finetuned_client_id = 0

    while finetuned_client_id < num_clients:
        logging.info("=" * 100)
        logging.info(f"Finetuning client {finetuned_client_id}")


        dataset = federated_dataset.get_task_dataset(task_id=finetuned_client_id, mode=args.split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        client_messages_metadata = {
            "global": all_messages_metadata["global"],
            "local": all_messages_metadata[f"{finetuned_client_id}"]
        }

        if args.task_name == "purchase":
            metric = multiclass_accuracy
        elif args.task_name == "adult":
            metric = binary_accuracy_with_sigmoid
        elif args.task_name == "purchase_binary":
            metric = binary_accuracy_with_sigmoid
        elif args.task_name == "medical_cost":
            metric = mean_squared_error
        elif args.task_name == "income":
            metric = mean_squared_error
        elif args.task_name == "toy_regression":
            metric = mean_squared_error
        else:
            raise NotImplementedError(
                f"Metric for task '{args.task_name}' is not implemented"
            )

        finetuning_trainer = initialize_finetuning_trainer(
            args=args,
            client_messages_metadata=client_messages_metadata,
            model_init_fn=model_init_fn,
            criterion=criterion,
            metric=metric,
            is_binary_classification=is_binary_classification,
        )

        for step in range(args.num_epochs):
            # TODO: this should be integrated in run_simulation.py
            if args.noise_factor is not None:
                noise_model = model_init_fn().to(args.device)
                if args.verbose:
                    model_norm = torch.linalg.norm(finetuning_trainer.get_param_tensor())
                    logging.info(f'Norm of the model {model_norm}')
                    noise_norm = torch.linalg.norm(get_param_tensor(noise_model) * args.noise_factor)
                    logging.info(f'Norm of the noise {noise_norm}')
                finetuning_trainer.model = add_noise(finetuning_trainer.model, noise_model, args.noise_factor)

            loss, metric = finetuning_trainer.fit_epoch(loader=dataloader)

            if step % args.save_freq == 0:

                os.makedirs(os.path.join(args.finetuned_models_dir, f"{finetuned_client_id}"), exist_ok=True)
                path = os.path.join(os.path.join(args.finetuned_models_dir, f"{finetuned_client_id}", f"{step}.pt"))
                path = os.path.abspath(path)
                finetuning_trainer.save_checkpoint(path)
                all_finetuned_models_metadata_dict[f"{finetuned_client_id}"][f"{step}"] = path

                if finetuning_trainer.lr_scheduler is not None:
                    finetuning_trainer.lr_scheduler.step()

            logging.info("+" * 50)
            logging.info(f"Task ID: {finetuned_client_id}")
            logging.info(f"Train Loss: {loss:.4f} | Train Metric: {metric:.4f} |")
            logging.info("+" * 50)

        last_saved_iteration = max(all_finetuned_models_metadata_dict[f"{finetuned_client_id}"], key=int)

        final_finetuned_models_metadata_dict[f"{finetuned_client_id}"] = (
            all_finetuned_models_metadata_dict[f"{finetuned_client_id}"][last_saved_iteration]
        )

        logging.info("Local model finetuned successfully.")

        finetuned_client_id += 1
        pbar.update(1)
        if args.compute_single_client:
            finetuned_client_id = num_clients

    pbar.close()

    all_finetuned_models_metadata_dict = swap_dict_levels(all_finetuned_models_metadata_dict)

    trajectory_path = os.path.join(args.metadata_dir, "finetuning_trajectory.json")
    with open(trajectory_path, "w") as f:
        json.dump(all_finetuned_models_metadata_dict, f)

    logging.info(f"The finetuned models have been saved successfully in {trajectory_path} .")

    logging.info("="*100)
    logging.info("Saving final models metadata...")


    with open(os.path.join(args.metadata_dir, "finetuned.json"), "w") as f:
        json.dump(final_finetuned_models_metadata_dict, f)

    logging.info("The final finetuned models have been saved successfully.")


if __name__ == "__main__":
    main()
