import os
import json
import logging

from fedklearn.datasets.adult.adult import FederatedAdultDataset
from fedklearn.datasets.toy.toy import FederatedToyDataset


def none_or_float(value):
    """
    Helper function to convert 'none' to None or float values.
    """
    if value.lower() == 'none':
        return None
    else:
        return float(value)


def configure_logging(args):
    """
    Set up logging based on verbosity level
    """
    logging.basicConfig(level=logging.INFO - (args.verbose - args.quiet) * 10)


def load_dataset(task_name, data_dir, rng):
    """
    Load a federated dataset based on the specified task name.

    Args:
        task_name (str): Name of the task for which the dataset is to be loaded.
        data_dir (str): Directory where the dataset should be stored or loaded from.
        rng (RandomState): NumPy random number generator for reproducibility.

    Returns:
        FederatedDataset: Initialized federated dataset.

    Raises:
        NotImplementedError: If the dataset initialization for the specified task is not implemented.
    """
    if task_name == "adult":
        return FederatedAdultDataset(
            cache_dir=data_dir,
            download=False,
            rng=rng
        )
    elif task_name == "toy_regression" or task_name == "toy_classification":
        return FederatedToyDataset(
            cache_dir=data_dir,
            allow_generation=False,
            force_generation=False,
            rng=rng
        )
    else:
        raise NotImplementedError(
            f"Dataset initialization for task '{task_name}' is not implemented."
        )


def weighted_average(scores, n_samples):
    if len(scores) != len(n_samples):
        raise ValueError("The lengths of 'scores' and 'n_samples' must be the same.")

    weighted_sum = sum(score * n_sample for score, n_sample in zip(scores, n_samples))

    total_samples = sum(n_samples)

    weighted_avg = weighted_sum / total_samples

    return weighted_avg


def save_scores(scores_list, n_samples_list, results_path):
    avg_score = weighted_average(scores=scores_list, n_samples=n_samples_list)
    logging.info("=" * 100)
    logging.info(f"Average Score={avg_score:.3f}")

    logging.info("=" * 100)
    logging.info("Saving simulation results..")
    results = [{"score": score, "n_samples": n_samples} for score, n_samples in zip(scores_list, n_samples_list)]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f)

    logging.info(f"The results dictionary has been saved in {results_path}")