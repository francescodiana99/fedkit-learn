import argparse
import copy
import logging

from fedklearn.utils import get_param_tensor

import numpy as np

from torch.utils.data import DataLoader

from fedklearn.attacks.aia import AttributeInferenceAttack
from utils import *
from constants import *

from tqdm import tqdm

def parse_args(args_list=None):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task_name",
        type=str,
        choices=['adult', 'toy_regression', 'toy_classification'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification'.",
        required=True
    )

    parser.add_argument(
        "--attacked_round",
        type=int,
        help="Round to test",
        required=True
    )

    parser.add_argument(
        "--models_metadata_path",
        type=str,
        help="Path to the metadata file of the models",
        required=True
    )

    parser.add_argument(
        "--models_config_metadata_path",
        type=str,
        help="Path to the metadata file of the models' configurations",
        required=True
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
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=1024,
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
        "--temperature",
        type=float,
        help="non-negative scalar temperature used for Gumbel-Softmax distribution.",
        default=1.0
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="non-negative scalar, between 0 and 1, used as a threshold in the binary case.",
        default=0.5
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        help="Number of simulation rounds",
        default=100
    )

    parser.add_argument(
        "--reference_models_metadata_path",
        type=str,
        help="Path to the metadata file of the models obtained with the Local Model Reconstruction Attack",
        required=True
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

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory to cache data"
    )

    parser.add_argument(
        "--debug",
        help="Enable debug mode",
        action="store_true",
        default=False
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def evaluate_trainer(trainer, dataloader):
    evaluation_trainer = copy.deepcopy(trainer)
    evaluation_trainer.criterion = nn.BCEWithLogitsLoss(reduction='mean') if evaluation_trainer.is_binary_classification else trainer.criterion
    loss, metric = evaluation_trainer.evaluate_loader(dataloader)
    return loss, metric


def initialize_random_trainer(model, criterion, optimizer, is_binary_classification, metric, device):
    """Initialize a random trainer"""
    random_trainer = Trainer(model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            is_binary_classification=is_binary_classification)
    return random_trainer


def initialize_metadata_dict_from_checkpoint(metadata_path, round):
    """Retrieve the metadata of the models from a specific checkpoint"""

    models_metadata_dict = dict()

    with open(metadata_path, "r") as f:
        models_metadata = json.load(f)

    for model_id in models_metadata.keys():
        models_metadata_dict[model_id] = models_metadata[model_id][str(round)]

    return models_metadata_dict

def compute_scores(task_name, federated_dataset, sensitive_attribute, sensitive_attribute_type, split,
                   reference_trainers_dict, criterion, is_binary_classification, learning_rate, optimizer_name,
                   aia_initialization, aia_num_rounds, device, rng, torch_rng, batch_size, trainers_dict, debug=False,
                   random_trainer=None):

    logging.info(f"Simulate AIA")

    n_samples_list = []

    num_clients = len(trainers_dict) - 1

    global_model = trainers_dict["global"].model

    scores_per_client_dict = {
        "local": dict(),
        "reference": dict(),
        "global": dict(),
        "random": dict()
    }

    metrics_dict = {
        "global": dict(),
        "random": dict(),
        "reference": dict()
    }

    for attacked_client_id in tqdm(range(num_clients)):
        logging.info("=" * 100)
        logging.info(f"Simulating attack for {attacked_client_id}...")

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=split)

        for client in trainers_dict.keys():
            if client != "global":
                scores_per_client_dict["local"][client] = 0
                scores_per_client_dict["reference"][client] = 0
                scores_per_client_dict["global"][client] = 0
                scores_per_client_dict["random"][client] = 0

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
        dataloader = DataLoader(dataset, batch_size=batch_size)

        if debug:
            try:
                local_model = trainers_dict[f"{attacked_client_id}"].model
            except KeyError:
                local_model = trainers_dict[attacked_client_id].model
            local_model.eval()

            if random_trainer is None:
                raise ValueError("random_trainer cannot be None in debug mode")
            else:
                random_model = random_trainer.model
                random_model.eval()

            # local model evaluation
            local_model_loss, local_model_metric = evaluate_trainer(trainers_dict[f"{attacked_client_id}"], dataloader)
            logging.info(
                f"Local model {split} loss={local_model_loss:.3f}, Test {split} metric={local_model_metric:.3f}")

            local_attacked_weight = local_model.fc.weight[:, sensitive_attribute_id]
            logging.info(f"Printing attacked feature's weight: {local_attacked_weight}")
            aia_score_local = evaluate_aia(model=local_model, dataset=dataset,
                                           sensitive_attribute_id=sensitive_attribute_id,
                                           sensitive_attribute_type=sensitive_attribute_type,
                                           initialization=aia_initialization, device=device,
                                           num_iterations=aia_num_rounds, criterion=criterion,
                                           is_binary_classification=is_binary_classification,
                                           learning_rate=learning_rate,
                                           optimizer_name=optimizer_name, success_metric=success_metric, rng=rng,
                                           torch_rng=torch_rng, output_losses=True)
            scores_per_client_dict["local"][attacked_client_id] = aia_score_local
            logging.info(f"Score={aia_score_local:.3f} for client {attacked_client_id} with local model")
            logging.info("=" * 100)

            # random model evaluation
            random_model_loss, random_model_metric = evaluate_trainer(random_trainer, dataloader)
            logging.info(
                f"Random model {split} loss={random_model_loss:.3f}, Test {split} metric={random_model_metric:.3f}")
            metrics_dict["random"][attacked_client_id] = random_model_metric

            logging.info(f"Printing random model parameters {get_param_tensor(random_model)}")

            random_attacked_weight = random_model.fc.weight[:, sensitive_attribute_id]
            logging.info(
                f"Printing attacked feature's weight: {random_attacked_weight}")

            aia_score_random = evaluate_aia(model=random_model, dataset=dataset,
                                            sensitive_attribute_id=sensitive_attribute_id,
                                            sensitive_attribute_type=sensitive_attribute_type,
                                            initialization=aia_initialization, device=device,
                                            num_iterations=aia_num_rounds,
                                            criterion=criterion,
                                            is_binary_classification=is_binary_classification,
                                            learning_rate=learning_rate,
                                            optimizer_name=optimizer_name, success_metric=success_metric, rng=rng,
                                            torch_rng=torch_rng, output_losses=True)
            scores_per_client_dict["random"][attacked_client_id] = aia_score_random
            logging.info(f"Score={aia_score_random:.3f} for client {attacked_client_id} with random model")
            logging.info("=" * 100)


        try:
            reference_model = reference_trainers_dict[f"{attacked_client_id}"].model
        except KeyError:
            reference_model = reference_trainers_dict[attacked_client_id].model

        reference_model.eval()
        global_model.eval()

        global_loss, global_metric = evaluate_trainer(trainers_dict["global"], dataloader)
        logging.info(f"Global model {split} loss={global_loss:.3f}, Test {split} metric={global_metric:.3f}")

        if debug:
            logging.info(f"Printing global model parameters: {get_param_tensor(global_model)}")
            global_attacked_weight = global_model.fc.weight[:, sensitive_attribute_id]
            logging.info(f"Printing attacked feature's weight: {global_attacked_weight}")


        aia_score_global = evaluate_aia(model=global_model, dataset=dataset,
                                        sensitive_attribute_id=sensitive_attribute_id,
                                        sensitive_attribute_type=sensitive_attribute_type,
                                        initialization=aia_initialization, device=device, num_iterations=aia_num_rounds,
                                        criterion=criterion,
                                        is_binary_classification=is_binary_classification, learning_rate=learning_rate,
                                        optimizer_name=optimizer_name, success_metric=success_metric, rng=rng,
                                        torch_rng=torch_rng, output_losses=True)
        logging.info(f"Score={aia_score_global:.3f} for client {attacked_client_id} with global model")
        logging.info("=" * 100)

        reference_model_loss, reference_model_metric = evaluate_trainer(
            trainer=reference_trainers_dict[f"{attacked_client_id}"],
            dataloader=dataloader
        )
        logging.info(
            f"Reference model {split} loss={reference_model_loss:.3f}, Test {split} metric={reference_model_metric:.3f}"
        )

        if debug:
            logging.info(f"Printing reference model parameters: {get_param_tensor(reference_model)}")
            reference_attacked_weight = reference_model.fc.weight[:, sensitive_attribute_id]
            logging.info(
                f"Printing attacked feature's weight: {reference_attacked_weight}")

        aia_score_reference = evaluate_aia(model=reference_model, dataset=dataset, num_iterations=aia_num_rounds,
                                           sensitive_attribute_id=sensitive_attribute_id,
                                           sensitive_attribute_type=sensitive_attribute_type,
                                           initialization=aia_initialization, device=device, criterion=criterion,
                                           is_binary_classification=is_binary_classification,
                                           learning_rate=learning_rate,
                                           optimizer_name=optimizer_name,
                                           success_metric=success_metric, rng=rng, torch_rng=torch_rng,
                                           output_losses=True)
        logging.info(f"Score={aia_score_reference:.3f} for client {attacked_client_id} with reference model")

        metrics_dict["reference"][attacked_client_id] = reference_model_metric
        metrics_dict["global"][attacked_client_id] = global_metric

        scores_per_client_dict["reference"][attacked_client_id] = aia_score_reference
        scores_per_client_dict["global"][attacked_client_id] = aia_score_global

        n_samples_list.append(len(dataset))

    return scores_per_client_dict, metrics_dict, n_samples_list


def main():

    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)

    with open(args.reference_models_metadata_path, "r") as f:
        reference_models_metadata_dict = json.load(f)

    with open(args.models_config_metadata_path, "r") as f:
        model_config_metadata_dict = json.load(f)

    criterion, model_init_fn, is_binary_classification, metric = get_trainer_parameters(task_name=args.task_name,
                                                                                        device=args.device,
                                                                                        model_config_path=
                                                                                        model_config_metadata_dict[
                                                                                            "model_config"])

    models_metadata_dict = initialize_metadata_dict_from_checkpoint(args.models_metadata_path, args.attacked_round)
    trainers_dict = initialize_trainers_dict(
        models_metadata_dict, criterion=criterion, model_init_fn=model_init_fn,
        is_binary_classification=is_binary_classification, metric=metric, device=args.device
    )

    if args.debug:
        random_trainer = initialize_random_trainer(model=model_init_fn(), criterion=criterion, metric=metric,
                                                   is_binary_classification=is_binary_classification, optimizer=None,
                                                   device=args.device)
    else:
        random_trainer = None

    reference_trainers_dict = initialize_trainers_dict(
        reference_models_metadata_dict, criterion=criterion, model_init_fn=model_init_fn,
        is_binary_classification=is_binary_classification, metric=metric, device=args.device
    )

    scores_per_client_dict, metrics_dict, n_samples_list = compute_scores(
        task_name=args.task_name,
        federated_dataset=federated_dataset,
        sensitive_attribute=args.sensitive_attribute,
        sensitive_attribute_type=args.sensitive_attribute_type,
        split=args.split,
        batch_size=args.batch_size,
        reference_trainers_dict=reference_trainers_dict,
        trainers_dict=trainers_dict,
        random_trainer=random_trainer,
        criterion=criterion,
        is_binary_classification=is_binary_classification,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        aia_initialization=args.initialization,
        aia_num_rounds=args.num_rounds,
        device=args.device,
        rng=rng, torch_rng=torch_rng, debug=args.debug
    )

    global_scores = list(metrics_dict["global"].values())
    avg_global_score = weighted_average(global_scores, n_samples_list)
    logging.info(f"Average metric for global model: {avg_global_score:.3f}")


    if args.debug:
        random_scores = list(metrics_dict["random"].values())
        avg_random_score = weighted_average(random_scores, n_samples_list)
        logging.info(f"Average metric for random model: {avg_random_score:.3f}")


    logging.info("Saving scores..")
    os.makedirs(args.results_dir, exist_ok=True)
    scores_path = os.path.join(args.results_dir, f"aia_round_{args.attacked_round}.json")
    with open(scores_path, "w") as f:
        json.dump(scores_per_client_dict, f)


if __name__ == "__main__":
    main()
