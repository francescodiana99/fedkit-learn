import os
import json
import logging


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
