import argparse
import copy
import logging

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from torch.utils.data import DataLoader

from fedklearn.utils import get_param_tensor

from miscellaneous.aia_dataset import AIADataset
from fedklearn.attacks.aia import AttributeInferenceAttack
from utils import *
from constants import *

from tqdm import tqdm

def parse_args(args_list=None):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--task_name",
        type=str,
        choices=['adult', 'toy_regression', 'toy_classification', 'purchase'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification', 'purchase'.",
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
        help="Path to the metadata file of the models obtained with the client Model Reconstruction Attack",
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
        "--save_aia_split",
        action="store_true",
        help="Save the data used for the AIA attack"
    )

    parser.add_argument(
        "--aia_data_path",
        type=str,
        help="Path to save the data split based on the AIA"
    )



    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def evaluate_on_aia_results(trainer, path):
    """Evaluate the model on the clones' features with higher loss. Used for debugging purposes"""
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dataset = AIADataset(df)
    dataloader = DataLoader(dataset, batch_size=1)
    loss, metric, all_losses = evaluate_trainer(trainer, dataloader)
    return loss, metric, all_losses

def evaluate_trainer(trainer, dataloader):
    evaluation_trainer = copy.deepcopy(trainer)
    if evaluation_trainer.is_binary_classification:
        evaluation_trainer.criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        evaluation_trainer.criterion = nn.CrossEntropyLoss(reduction='mean')
    avg_loss, metric = evaluation_trainer.evaluate_loader(dataloader, output_losses=False)
    return avg_loss, metric

def initialize_random_trainer(model, criterion, optimizer, is_binary_classification, metric, device):
    """Initialize a random trainer"""
    random_trainer = DebugTrainer(model=model,
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
    trainers_dict, reference_trainers_dict, criterion, is_binary_classification, learning_rate, optimizer_name,
    aia_initialization, aia_num_rounds, device, rng, torch_rng, batch_size, random_trainer):

    logging.info(f"Simulate AIA")

    n_samples_list = []

    num_clients = len(trainers_dict) - 1

    global_model = trainers_dict["global"].model
    # DEBUG
    # global_model.fc.weight.data[:,39] = 0.0

    random_model = random_trainer.model

    scores_per_client_dict = {
        "client": dict(),
        "reference": dict(),
        "global": dict(),
        "random": dict()
    }

    metrics_dict = {
        "client": dict(),
        "reference": dict(),
        "global": dict(),
        "random": dict()
    }

    for attacked_client_id in tqdm(range(num_clients)):
        logging.info("=" * 100)
        logging.info(f"Simulating attack for {attacked_client_id}...")

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=split)

        if task_name == "adult" or task_name == "purchase":
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
            client_model = trainers_dict[f"{attacked_client_id}"].model
        except KeyError:
            client_model = trainers_dict[attacked_client_id].model

        try:
            reference_model = reference_trainers_dict[f"{attacked_client_id}"].model
            # DEBUG
            # reference_model.fc.weight.data[:,39] = 0.0

        except KeyError:
            reference_model = reference_trainers_dict[attacked_client_id].model
            # DEBUG
            # reference_model.fc.weight.data[:,39] = 0.1

        client_model.eval()
        reference_model.eval()
        global_model.eval()
        random_model.eval()

        # Evaluate the models before the attack
        dataloader = DataLoader(dataset, batch_size=batch_size)
        global_loss, global_metric = evaluate_trainer(trainers_dict["global"], dataloader)
        client_model_loss, client_model_metric = evaluate_trainer(trainers_dict[f"{attacked_client_id}"], dataloader)
        reference_model_loss, reference_model_metric = evaluate_trainer(reference_trainers_dict[f"{attacked_client_id}"], dataloader)
        random_model_loss, random_model_metric = evaluate_trainer(random_trainer, dataloader)

        logging.info(f"Global model {split} loss={global_loss:.3f},  {split} metric={global_metric:.3f}")
        logging.info(f"client model {split} loss={client_model_loss:.3f},  {split} metric={client_model_metric:.3f}")
        logging.info(f"reference model {split} loss={reference_model_loss:.3f},  {split} metric={reference_model_metric:.3f}")
        logging.info(f"Random model {split} loss={random_model_loss:.3f},  {split} metric={random_model_metric:.3f}")

        metrics_dict["client"][attacked_client_id] = client_model_metric
        metrics_dict["reference"][attacked_client_id] = reference_model_metric
        metrics_dict["global"][attacked_client_id] = global_metric
        metrics_dict["random"][attacked_client_id] = random_model_metric

        logging.info("Attack with the CLIENT model...")
        aia_score_client = evaluate_aia(model=client_model, dataset=dataset,
            sensitive_attribute_id=sensitive_attribute_id, sensitive_attribute_type=sensitive_attribute_type,
            initialization=aia_initialization, device=device, num_iterations=aia_num_rounds, criterion=criterion,
            is_binary_classification=is_binary_classification, learning_rate=learning_rate,
            optimizer_name=optimizer_name, success_metric=success_metric, rng=rng, torch_rng=torch_rng,
            output_predictions=True)

        logging.info("Attack with the REFERENCE model...")
        aia_score_reference = evaluate_aia(model=reference_model, dataset=dataset, num_iterations=aia_num_rounds,
            sensitive_attribute_id=sensitive_attribute_id, sensitive_attribute_type=sensitive_attribute_type,
            initialization=aia_initialization, device=device, criterion=criterion,
            is_binary_classification=is_binary_classification, learning_rate=learning_rate,
            optimizer_name=optimizer_name, success_metric=success_metric, rng=rng, torch_rng=torch_rng,
                                           output_predictions=True)

        logging.info("Attack with the GLOBAL model...")
        aia_score_global = evaluate_aia(model=global_model, dataset=dataset,
                                        sensitive_attribute_id=sensitive_attribute_id,
                                        sensitive_attribute_type=sensitive_attribute_type,
                                        initialization=aia_initialization, device=device, num_iterations=aia_num_rounds,
                                        criterion=criterion,
                                        is_binary_classification=is_binary_classification, learning_rate=learning_rate,
                                        optimizer_name=optimizer_name, success_metric=success_metric, rng=rng,
                                        torch_rng=torch_rng, output_predictions=True)

        logging.info("Attack with the RANDOM model...")
        aia_score_random = evaluate_aia(model=random_model, dataset=dataset,
                                        sensitive_attribute_id=sensitive_attribute_id,
                                        sensitive_attribute_type=sensitive_attribute_type,
                                        initialization=aia_initialization, device=device, num_iterations=aia_num_rounds,
                                        criterion=criterion,
                                        is_binary_classification=is_binary_classification, learning_rate=learning_rate,
                                        optimizer_name=optimizer_name, success_metric=success_metric, rng=rng,
                                        torch_rng=torch_rng, output_predictions=True)

        logging.info(f"global model parameters: {get_param_tensor(global_model)}")
        logging.info(f"reference model parameters: {get_param_tensor(reference_model)}")

        logging.info(f"Score={aia_score_client:.3f} for client {attacked_client_id} with client model")
        logging.info(f"Score={aia_score_reference:.3f} for client {attacked_client_id} with reference model")
        logging.info(f"Score={aia_score_global:.3f} for client {attacked_client_id} with global model")
        logging.info(f"Score={aia_score_random:.3f} for client {attacked_client_id} with random model")

        scores_per_client_dict["client"][attacked_client_id] = aia_score_client
        scores_per_client_dict["reference"][attacked_client_id] = aia_score_reference
        scores_per_client_dict["global"][attacked_client_id] = aia_score_global
        scores_per_client_dict["random"][attacked_client_id] = aia_score_random

        n_samples_list.append(len(dataset))

    return scores_per_client_dict, metrics_dict, n_samples_list

def box_plot(all_losses_flip, all_losses_no_flip, model_name):
    data = [all_losses_flip, all_losses_no_flip]

    plt.figure(figsize=(16, 12))
    labels = ['Flip', 'No Flip']
    plt.xticks(range(len(labels)), labels)
    plt.title(model_name)
    sns.boxplot(data=data)

    # Add labels and title
    plt.ylabel('Loss')
    # Show plot
    plt.show()


def log_metrics(trainer, path):
    for attacked_client_id in range(len(trainer)):
        flip_features_loss, flip_features_metric, all_losses_flip = evaluate_on_aia_results(
            trainer[f"{attacked_client_id}"], path=os.path.join(path, f"{attacked_client_id}", "flipped_features.csv"))

        wrong_aia_flip_loss, wrong_aia_flip_metric, wrong_aia_flip_all_losses = evaluate_on_aia_results(
            trainer[f"{attacked_client_id}"], path=os.path.join(path,f"{attacked_client_id}", "recon_error_flipped_feature.csv"))
        wrong_aia_correct_features_loss, wrong_aia_correct_features_metric, all_losses_wrong_aia_correct_features = (
            evaluate_on_aia_results(
            trainer[f"{attacked_client_id}"], path=os.path.join(path, f"{attacked_client_id}", "recon_error_initial_feature.csv")))

        correct_predictions_loss, correct_predictions_metric, all_losses_correct_preds = evaluate_on_aia_results(
            trainer[f"{attacked_client_id}"], path=os.path.join(path, f"{attacked_client_id}", "correct_reconstructions.csv"))

        right_recon_flip_feature_loss, right_recon_flip_feature_metric, all_losses_right_recon_flip_feature =\
            evaluate_on_aia_results(
            trainer[f"{attacked_client_id}"], path=os.path.join(path, f"{attacked_client_id}", "correct_recon_flipped_feature.csv"))

        logging.info(f"Model {attacked_client_id} loss all flip: {flip_features_loss:.3f},"
                     f" accuracy: {flip_features_metric:.3f}")

        logging.info(f" Model {attacked_client_id} loss Success/No Flip: "
                     f"{correct_predictions_loss:.3f}, accuracy: "
                     f"{correct_predictions_metric:.3f}")
        logging.info(f" Model {attacked_client_id} loss Success/Flip: "
                     f"{right_recon_flip_feature_loss:.3f}, "
                     f"accuracy: {right_recon_flip_feature_metric:.3f}")
        logging.info(
            f" Model {attacked_client_id} loss on Fail/Flip: {wrong_aia_flip_loss:.3f}, "
            f"accuracy: {wrong_aia_flip_metric:.3f}")
        logging.info(f"Model {attacked_client_id} loss Fail/No Flip:"
                     f" {wrong_aia_correct_features_loss:.3f}, "
                     f"accuracy: {wrong_aia_correct_features_metric:.3f}")


def main():

    args = parse_args()

    configure_logging(args)

    rng = np.random.default_rng(seed=args.seed)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)

    models_metadata_dict = initialize_metadata_dict_from_checkpoint(args.models_metadata_path, args.attacked_round)

    with open(args.models_config_metadata_path, "r") as f:
        model_config_metadata_dict = json.load(f)

    with open(args.reference_models_metadata_path, "r") as f:
        reference_models_metadata_dict = json.load(f)

    criterion, model_init_fn, is_binary_classification, metric = get_trainer_parameters(
        task_name=args.task_name, device=args.device,
        model_config_path=model_config_metadata_dict["model_config"]
    )

    trainers_dict = initialize_trainers_dict(
        models_metadata_dict, criterion=criterion, model_init_fn=model_init_fn,
        is_binary_classification=is_binary_classification, metric=metric, device=args.device
    )

    reference_trainers_dict = initialize_trainers_dict(
        reference_models_metadata_dict, criterion=criterion, model_init_fn=model_init_fn,
        is_binary_classification=is_binary_classification, metric=metric, device=args.device
    )

    random_trainer = initialize_random_trainer(model=model_init_fn(), criterion=criterion, metric=metric,
                                               is_binary_classification=is_binary_classification, optimizer=None,
                                               device=args.device)

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
        rng=rng, torch_rng=torch_rng
    )

    global_scores = list(scores_per_client_dict["global"].values())
    global_metric = list(metrics_dict["global"].values())
    avg_global_score = weighted_average(global_scores, n_samples_list)
    avg_global_metric = weighted_average(global_metric, n_samples_list)

    random_scores = list(metrics_dict["random"].values())
    random_metric = list(scores_per_client_dict["random"].values())
    avg_random_score = weighted_average(random_scores, n_samples_list)
    avg_random_metric = weighted_average(random_metric, n_samples_list)

    reference_scores = list(scores_per_client_dict["reference"].values())
    reference_metric = list(metrics_dict["reference"].values())
    avg_reference_score = weighted_average(reference_scores, n_samples_list)
    avg_reference_metric = weighted_average(reference_metric, n_samples_list)


    logging.info(f"Average metric for global model: {avg_global_metric:.3f}")
    logging.info(f"Average metric for random model: {avg_random_metric:.3f}")
    logging.info(f"Average metric for reference model: {avg_reference_metric:.3f}")

    logging.info(f"Average score for global model: {avg_global_score:.3f}")
    logging.info(f"Average score for random model: {avg_random_score:.3f}")
    logging.info(f"Average score for reference model: {avg_reference_score:.3f}")



if __name__ == "__main__":
    main()
