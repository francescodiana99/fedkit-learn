import copy


def weighted_average(models, weights):
    """
    Compute the weighted average of a list of PyTorch models.

    Parameters:
    - models (list): List of PyTorch models.
    - weights (list): List of weights corresponding to each model.

    Returns:
    - model: PyTorch model representing the weighted average.
    """
    assert len(models) == len(weights), "Number of models and weights must be the same."

    # Initialize the weighted average model
    avg_model = copy.deepcopy(models[0])

    for avg_param, *model_params in zip(avg_model.parameters(), *[model.parameters() for model in models[1:]]):
        avg_param.data.mul_(weights[0])
        for weight, param in zip(weights[1:], model_params):
            avg_param.data.add_(weight * param.data)
        avg_param.data.div_(sum(weights))

    return avg_model
