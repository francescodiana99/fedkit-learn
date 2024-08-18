"""Attribute Inference Attack Simulation Script

Implementation based on the technique present in (https://arxiv.org/abs/2108.06910)_
"""

import argparse
import logging

import numpy as np

from tqdm import tqdm
import time


from torch.utils.tensorboard import SummaryWriter

from fedklearn.attacks.aia import AttributeInferenceAttack

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
        choices=['adult', 'toy_regression', 'toy_classification', 'purchase', 'purchase_binary', 'medical_cost',
                 'income', 'binary_income', 'linear_income', 'linear_medical_cost'],
        help="Task name. Possible are: 'adult', 'toy_regression', 'toy_classification, 'purchase', 'purchase_binary, "
             "'medical_cost', 'income', 'binary_income', 'linear_income', 'linear_medical_cost'.",
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
        "--keep_rounds_frac",
        type=float,
        help="Fraction of rounds to keep."
             "If set to 0.0, all rounds, except the last, will be discarded."
             "If set to 1.0, all rounds will be kept."
             "If set to a value between 0.0 and 1.0, it determines the fraction of rounds to keep "
             "starting from the end of the list. Defaults to 0. (i.e., discarding all rounds, except the last)."
             "If '--keep_first_rounds' is set to True:"
             "If set to 0.0, all rounds, except the first, will be discarded."
             "If set to 1.0, all rounds will be kept."
             "If set to a value between 0.0 and 1.0, it determines the fraction of rounds to keep "
             "starting from the beginning of the list.",
        default=0.0
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
        "--local_models_metadata_path",
        type=str,
        help="Path to the file containing the metadata of the local models.",
        default=None
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

    parser.add_argument(

        "--keep_first_rounds",
        default=False,
        action="store_true",
        help="If set, the fraction of round to keep will be determined "
             "starting from the beginning of the list of rounds. Default is False.")

    parser.add_argument(

        "--compute_single_client",
        help="If set, the attack will be computed just for the first client.",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--attacked_task",
        type=int,
        default=None,
        help="If set, the attack will be computed just for the specified client."
    )

    parser.add_argument(
        "--flip_percentage",
        type=float,
        default=0.0,
        help="Percentage of the sensitive attribute to flip")

    parser.add_argument(
        "--test",
        default=False,
        action="store_true")

    parser.add_argument(
        "--active_server",
        default=False,
        action="store_true",
        help='If set, it will simulate an active server and instead of using the clients and global messages, it will'
             'use the difference between local models at each round to simulate the attack. Note that this simulates the '
             'case of 1 local epoch.'
    )

    parser.add_argument(
        "--isolated",
        default=False,
        action="store_true",
        help='If set, it will simulate an active server that isolates a task by sending back the received model'
    )

    parser.add_argument(
        "--local_steps",
        type=int,
        help="Number of local steps. Used only when the attack is isolated.",
        default=1
    )

    parser.add_argument(
        "--active_adam",
        default=False,
        action="store_true",
        help='If set, it will simulate an active server with Adam server'
    )

    parser.add_argument(
        "--track_time",
        action="store_true",
        default=False,
        help="If set, it will track the time of the attack."
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def main():
    args = parse_args()

    configure_logging(args)
    set_seeds(args.seed)

    rng = np.random.default_rng(seed=args.seed)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed)

    federated_dataset = load_dataset(task_name=args.task_name, data_dir=args.data_dir, rng=rng)
    # TODO: fix this, should take only the clients in the metadata dict
    num_clients = len(federated_dataset.task_id_to_name)

    with open(args.metadata_path, "r") as f:
        all_messages_metadata = json.load(f)
    if args.active_server:
        if args.local_models_metadata_path is not None:
            with open(args.local_models_metadata_path, "r") as f:
                local_models_metadata = json.load(f)
            if args.isolated:
                n_rounds = len(local_models_metadata.keys())
                communication_rounds = [f"{i}" for i in range(0, n_rounds, args.local_steps)]

                if args.keep_first_rounds:
                    keep_round_ids = get_first_rounds(communication_rounds,
                                                      keep_frac=args.keep_rounds_frac)
                else:
                    keep_round_ids = get_last_rounds(communication_rounds,
                                                     keep_frac=args.keep_rounds_frac)
            elif args.active_adam:
            # TODO: this might be coded better
                n_rounds = len(local_models_metadata['0'].keys())
                communication_rounds = [f"{i}" for i in range(0, n_rounds, args.local_steps)]
                if args.keep_first_rounds:
                    keep_round_ids = get_first_rounds(communication_rounds,
                                                      keep_frac=args.keep_rounds_frac)
                else:
                    keep_round_ids = get_last_rounds(communication_rounds,
                                                     keep_frac=args.keep_rounds_frac)

        else:
            raise ValueError("The local models metadata path is required when  'active server' option is set.")

    else:
        if args.keep_first_rounds:
            keep_round_ids = get_first_rounds(all_messages_metadata["global"].keys(), keep_frac=args.keep_rounds_frac)
        else:
            keep_round_ids = get_last_rounds(all_messages_metadata["global"].keys(), keep_frac=args.keep_rounds_frac)

    # TODO: fix, it is not a binary classification but we need the shape transformation
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
    elif args.task_name == "linear_medical_cost":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "income":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "binary_income":
        criterion = nn.BCEWithLogitsLoss().to(args.device)
        is_binary_classification = True
    elif args.task_name == "linear_income":
        criterion = nn.MSELoss().to(args.device)
        is_binary_classification = True
    else:
        raise NotImplementedError(
            f"Network initialization for task '{args.task_name}' is not implemented"
        )
    with open(os.path.join(os.path.dirname(args.metadata_path), "model_config.json"), "r") as f:
        model_config_path = json.load(f)
    model_init_fn = lambda: initialize_model(model_config_path["model_config"])

    scores_list = []
    n_samples_list = []

    logging.info("Simulate Attacks..")

    all_clients_scores = []
    all_clients_cos_dis = []
    all_clients_l2_dis = []
    all_clients_att_time = []

    pbar = tqdm(range(num_clients))
    attacked_client_id = 0

    if args.attacked_task is not None:
        attacked_client_id = args.attacked_task
        args.compute_single_client = True

    while attacked_client_id < num_clients:

        logging.info("=" * 100)
        logging.info(f"Simulating attack for {attacked_client_id}...")

        logger = SummaryWriter(os.path.join(args.logs_dir, f"{attacked_client_id}"))

        dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=args.split)

        if args.task_name in ["adult", "purchase", "purchase_binary", "medical_cost", "income", "binary_income",
                              "linear_income", "linear_medical_cost"]:
            sensitive_attribute_id = dataset.column_name_to_id[args.sensitive_attribute]
            sensitive_attribute_type = args.sensitive_attribute_type
        elif args.task_name == "toy_classification" or args.task_name == "toy_regression":
            sensitive_attribute_id = federated_dataset.sensitive_attribute_id
            sensitive_attribute_type = federated_dataset.sensitive_attribute_type
        else:
            raise NotImplementedError(
                f"Dataset initialization for task '{args.task_name}' is not implemented."
            )

        success_metric = threshold_binary_accuracy if sensitive_attribute_type == "binary" else mean_squared_error
        if args.active_server and args.isolated:
            client_messages_metadata = get_active_messages_metadata(local_models_metadata=local_models_metadata,
                                                                    attacked_client_id=f"{attacked_client_id}",
                                                                    keep_round_ids=keep_round_ids,
                                                                    rounds_frac=args.keep_rounds_frac,
                                                                    use_isolate=True)
        elif args.active_server and args.active_adam:
            client_messages_metadata = get_active_messages_metadata(local_models_metadata=local_models_metadata,
                                                                      attacked_client_id=f"{attacked_client_id}",
                                                                      keep_round_ids=keep_round_ids,
                                                                      rounds_frac=args.keep_rounds_frac,
                                                                      use_isolate=False)
        else:
            client_messages_metadata = {
                "global": {key: all_messages_metadata["global"][key] for key in keep_round_ids},
                "local": {key: all_messages_metadata[f"{attacked_client_id}"][key] for key in keep_round_ids}
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
            torch_rng=torch_rng,
            flip_percentage=args.flip_percentage,
            test=args.test
        )
        if args.track_time:
            start_time = time.time()
        all_cos_dis, all_l2_dist = attack_simulator.execute_attack(num_iterations=args.num_rounds, output_losses=True)
        # TODO: remove this, only to check the results
        predictions = (attack_simulator.predicted_features)
        true_labels = (attack_simulator.true_labels)
        if args.track_time:
            end_time = time.time()
            logging.info(f"Attack time for client {attacked_client_id}: {end_time - start_time:.3f} seconds")
            all_clients_att_time.append(end_time - start_time)

        cos_dis = all_cos_dis[-1]
        l2_dist = all_l2_dist[-1]


        score = attack_simulator.evaluate_attack()
        all_clients_scores.append(score)
        all_clients_cos_dis.append(cos_dis/ len(keep_round_ids))
        all_clients_l2_dis.append(l2_dist)
        logging.info(f"Score={score:.3f} for client {attacked_client_id}")
        logging.info(f"L2 distance={l2_dist:.3f} for client {attacked_client_id}")

        scores_list.append(score)
        n_samples_list.append(len(dataset))

        attacked_client_id += 1
        pbar.update(1)
        if args.compute_single_client:
            attacked_client_id = num_clients

    pbar.close()

    avg_score = weighted_average(all_clients_scores, n_samples_list)
    avg_cos_dis = weighted_average(all_clients_cos_dis, n_samples_list)
    avg_l2_dis = weighted_average(all_clients_l2_dis, n_samples_list)
    logging.info(f"Average score: {avg_score}")
    logging.info(f"Average cosine dissimilarity: {avg_cos_dis}")
    logging.info(f"Average L2 distance: {avg_l2_dis}")

    # TODO: remove this, only to check the results
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)
    with open("true_labels.json", "w") as f:
        json.dump(true_labels, f)

    save_scores(scores_list=scores_list, n_samples_list=n_samples_list, results_path=args.results_path)
    save_scores(scores_list=all_clients_cos_dis, n_samples_list=n_samples_list,
                results_path=args.results_path.replace(".json", "_cos_dis.json"))
    save_scores(scores_list=all_clients_l2_dis, n_samples_list=n_samples_list,
                results_path=args.results_path.replace(".json", "_l2_dis.json"))
    if args.track_time:
        time_dict = dict()
        time_dict["results"] =  [{"time": time, "n_samples": n_samples} for time, n_samples in zip(all_clients_att_time, n_samples_list)]
        time_dict["device"] = get_gpu()
        with open(args.results_path.replace(".json", "_time.json"), "w") as f:
            json.dump(time_dict, f)

    logging.info(f"Results saved in {args.results_path}")
if __name__ == "__main__":
    main()
