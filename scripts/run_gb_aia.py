"""Gradient Based Attribute Inference Attack Simulation Script

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
        "--keep_rounds_fracs",
        nargs='+',
        type=float,
        help="List of fraction of rounds to keep and test."
             "If the single value is set to 0.0, all rounds, except the last, will be discarded."
             "If set to 1.0, all rounds will be kept."
             "If set to a value between 0.0 and 1.0, it determines the fraction of rounds to keep "
             "starting from the end of the list. Defaults to 0. (i.e., discarding all rounds, except the last)."
             "If '--keep_first_rounds' is set to True:"
             "If set to 0.0, all rounds, except the first, will be discarded."
             "If set to 1.0, all rounds will be kept."
             "If set to a value between 0.0 and 1.0, it determines the fraction of rounds to keep "
             "starting from the beginning of the list."
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
        "--metadata_dir",
        type=str,
        help="Metadata directory path",
        required=True
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
        "--learning_rates",
        nargs='+',
        type=float,
        default=1e-2,
        help="List of learning rates to test. If a single value is set, it will be used for all the experiments."
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
        "--attacked_round",
        type=int,
        default=None,
        help="First round to be attacked when considering an active attacker."
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
        "--attacked_task",
        type=int,
        default=None,
        help="If set, the attack will be computed just for the specified client."
    )

    parser.add_argument(
        "--active_server",
        default=False,
        action="store_true",
        help='If set, it will simulate an active server and instead of using the clients and global messages, it will'
             'use the difference between local models at each round to simulate the attack.'
    )

    parser.add_argument(
        "--track_time",
        action="store_true",
        default=False,
        help="If set, it will track the execution time of the attack in seconds."
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

    try:
        with open(os.path.join(args.metadata_dir, "setup.json"), "r") as f:
            fl_setup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Federated Learning simulation metadata file not found at \
                                '{args.metadata_dir}/setup.json'.")
    
    federated_dataset = load_dataset(fl_setup, rng=rng)
    num_clients = len(federated_dataset.task_id_to_name)

    if args.active_server:

        isolated_metadata_path = os.path.join(args.metadata_dir, f"isolated_{args.attacked_round}.json")
        with open(isolated_metadata_path, "r") as f:
            isolated_models_metadata = json.load(f)
        
        n_rounds = len(isolated_models_metadata.keys())
        communication_rounds = [f"{i}" for i in range(0, n_rounds, fl_setup["local_steps"])]

    else:
        metadata_path = os.path.join(args.metadata_dir, "federated.json")
        with open(metadata_path, "r") as f:
            all_messages_metadata = json.load(f)
        n_rounds = len(all_messages_metadata["global"].keys())
        communication_rounds = list(all_messages_metadata["global"].keys())

    criterion, _, cast_float = get_trainers_config(fl_setup["task_name"])
    model_init_fn = lambda: initialize_model(fl_setup["model_config_path"])

    logging.info("Simulate Attacks..")

    for rounds_frac in args.keep_rounds_fracs:

        for learning_rate in args.learning_rates:
            logging.info("+" * 50)
            logging.info(f"Testing round fraction: {rounds_frac} and learning rate: {learning_rate}.")

            scores_list = []
            n_samples_list = []
            cos_dis_list = []
            l2_dist_list = []
            all_clients_att_time = []

            pbar = tqdm(range(num_clients))
            attacked_client_id = 0

            if args.keep_first_rounds:
                keep_round_ids = get_first_rounds(communication_rounds,
                                                    keep_frac=rounds_frac)
            else:
                keep_round_ids = get_last_rounds(communication_rounds,
                                                    keep_frac=rounds_frac)

            if args.attacked_task is not None:
                attacked_client_id = args.attacked_task
                compute_single_client = True
            else:
                compute_single_client = False

            while attacked_client_id < num_clients:

                logging.info("=" * 100)
                logging.info(f"Simulating attack for client {attacked_client_id}...")

                logger = SummaryWriter(os.path.join(args.logs_dir, f"{attacked_client_id}"))

                dataset = federated_dataset.get_task_dataset(task_id=attacked_client_id, mode=args.split)

                if fl_setup["task_name"] in ["adult", "purchase", "purchase_binary", "medical_cost", "income", "binary_income"]:
                    sensitive_attribute_id = dataset.column_name_to_id[args.sensitive_attribute]
                    sensitive_attribute_type = args.sensitive_attribute_type
                elif fl_setup["task_name"] == "toy_classification" or args.task_name == "toy_regression":
                    sensitive_attribute_id = federated_dataset.sensitive_attribute_id
                    sensitive_attribute_type = federated_dataset.sensitive_attribute_type
                else:
                    raise NotImplementedError(
                        f"Dataset initialization for task '{fl_setup["task_name"]}' is not implemented."
                    )

                success_metric = threshold_binary_accuracy if sensitive_attribute_type == "binary" else mean_squared_error
                if args.active_server:
                    client_messages_metadata = get_isolated_messages_metadata(isolated_models_metadata=isolated_models_metadata,
                                                                            attacked_client_id=f"{attacked_client_id}",
                                                                            keep_round_ids=keep_round_ids,
                                                                            rounds_frac=rounds_frac,
                                                                            use_isolate=True)
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
                    cast_float=cast_float,
                    learning_rate=learning_rate,
                    optimizer_name=args.optimizer,
                    success_metric=success_metric,
                    logger=logger,
                    log_freq=args.log_freq,
                    rng=rng,
                    torch_rng=torch_rng,
                )
                if args.track_time:
                    start_time = time.time()
                all_cos_dis, all_l2_dist = attack_simulator.execute_attack(num_iterations=args.num_rounds, output_losses=True)

                if args.track_time:
                    end_time = time.time()
                    logging.info(f"Attack time for client {attacked_client_id}: {end_time - start_time:.3f} seconds")
                    all_clients_att_time.append(end_time - start_time)

                cos_dis = all_cos_dis[-1]
                l2_dist = all_l2_dist[-1]
                score = attack_simulator.evaluate_attack()
                logging.info(f"Score={score:.3f} for client {attacked_client_id}")
                logging.info(f"L2 distance={l2_dist:.3f} for client {attacked_client_id}")

                scores_list.append(score)
                cos_dis_list.append(cos_dis/ len(keep_round_ids))
                l2_dist_list.append(l2_dist)
                n_samples_list.append(len(dataset))

                attacked_client_id += 1
                pbar.update(1)
                if compute_single_client:
                    attacked_client_id = num_clients
            pbar.close()

            avg_score = weighted_average(scores_list, n_samples_list)
            avg_cos_dis = weighted_average(cos_dis_list, n_samples_list)
            avg_l2_dis = weighted_average(l2_dist_list, n_samples_list)
            logging.info(f"Average score: {avg_score}")
            logging.info(f"Average cosine dissimilarity: {avg_cos_dis}")
            logging.info(f"Average L2 distance: {avg_l2_dis}")
            if args.track_time:
                logging.info(f"Total time in seconds: {sum(all_clients_att_time)} s.")
                time_dict = {"time": sum(all_clients_att_time), "device": get_device_info()}
                save_aia_gb_score(args.results_path, rounds_frac, learning_rate, avg_score, avg_cos_dis, avg_l2_dis, time_dict)
            else:
                save_aia_gb_score(args.results_path, rounds_frac, learning_rate, avg_score, avg_cos_dis, avg_l2_dis)
            logging.info("=" * 100)
            logging.info(f"Results successfully  savedin {args.results_path}.")

if __name__ == "__main__":
    main()
