import torch

import torch.nn as nn

from torch.overrides import has_torch_function_unary, handle_torch_function

from copy import deepcopy


def copy_model(target, source):
    """
    Copy the state dictionary from the source model to the target model.

    This function is used to synchronize the parameters of two PyTorch models by copying
    the state dictionary from the source model to the target model.

    Parameters
    ----------
    - target (nn.Module): The target PyTorch model to be updated.
    - source (nn.Module): The source PyTorch model from which the state dictionary is copied.
    """
    target.load_state_dict(source.state_dict())


def get_param_tensor(model):
    """
    Get the parameters of the `model` as a unique flattened tensor.

    Parameters
    ----------
    model (nn.Module): The PyTorch model from which parameters will be extracted.

    Returns
    -------
    torch.Tensor: A flattened tensor containing the concatenated parameters of the model.

    Notes
    -----
    This method iterates over all parameters of the model and flattens them into a single tensor.
    """
    param_list = []

    for param in model.parameters():
        param_list.append(param.data.view(-1, ))

    return torch.cat(param_list)


def set_param_tensor(model, param_tensor, device):
    """
    Set the parameters of `model` from the provided tensor. The operation is done in-place.

    Parameters
    ----------
    model (torch.nn.Module): The PyTorch model to be updated.

    param_tensor (torch.Tensor): Tensor of shape (`model_dim`, ) containing the parameters to
        update the model.

    device (torch.Device or str): The device to which the tensors should be moved.

    Notes
    -----
    This method iterates over all parameters of the model, extracts the corresponding portion
    from the provided tensor, and reshapes it to match the original parameter shape.
    """
    param_tensor = param_tensor.to(device)

    current_index = 0
    for param in model.parameters():
        param_shape = param.data.shape
        current_dimension = param.data.view(-1, ).shape[0]

        param.data = deepcopy(param_tensor[current_index: current_index + current_dimension].reshape(param_shape))

        current_index += current_dimension


def get_grad_tensor(model):
    """
    Get `model` gradients as a unique flattened tensor.

    Parameters
    ----------
    model (nn.Module): The PyTorch model from which gradients will be extracted.

    Returns
    -------
    torch.Tensor: Flattened tensor containing gradients of the model parameters.
    """

    grad_list = []

    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.data.view(-1, ))

    return torch.cat(grad_list)


def set_grad_tensor(model, grad_tensor, device):
    """
    Set the gradients of `model` from the provided tensor.

    Parameters
    ----------
    model (torch.nn.Module): The PyTorch model to be updated.

    grad_tensor (torch.Tensor): Tensor of shape (`model_dim`, ) containing gradients to update
        the model.

    device (torch.Device or str): The device to which the tensors should be moved.

    Notes
    -----
    This method iterates over all parameters of the model, extracts the corresponding portion
    from the provided tensor, and reshapes it to match the original parameter shape.
    """
    grad_tensor = grad_tensor.to(device)

    current_index = 0
    for param in model.parameters():
        param_shape = param.data.shape
        current_dimension = param.data.view(-1, ).shape[0]

        param.grad.data = deepcopy(grad_tensor[current_index: current_index + current_dimension].reshape(param_shape))

        current_index += current_dimension


def map_to_closest_scalar(tensor, scalar1, scalar2):
    """
    Map each element of the tensor to the closest among two scalars.

    Parameters:
        tensor (torch.Tensor): Input tensor.
        scalar1 (float): First scalar.
        scalar2 (float): Second scalar.

    Returns:
        torch.Tensor: Tensor with elements mapped to the closest scalar.
    """
    distance_to_scalar1 = torch.abs(tensor - scalar1)
    distance_to_scalar2 = torch.abs(tensor - scalar2)

    result_tensor = torch.where(distance_to_scalar1 < distance_to_scalar2, scalar1, scalar2)

    return result_tensor


def jsd(p, q, distribution_type, epsilon=1e-10):
    """
    Compute the Jensen-Shannon Divergence (JSD) between two probability distributions.

    Parameters:
    - p (torch.Tensor): Probability tensor for the first distribution.
    - q (torch.Tensor): Probability tensor for the second distribution.
    - distribution_type (str, optional): Type of probability distribution. Default is 'bernoulli'.
      Possible values: 'bernoulli', 'multinomial', 'gaussian'.

    Returns:
    - torch.Tensor: Jensen-Shannon Divergence between the two distributions.

    Raises:
    - ValueError: If an invalid distribution type is provided.

    Notes:
    - For 'bernoulli':
        - p, q should be tensors of binary values (0 or 1).
    - For 'multinomial':
        - p, q should be tensors representing probability vectors for discrete events.
    - For 'gaussian':
        - p, q should be tensors representing means of the Gaussian distributions.

    """
    # TODO: check formulas
    if distribution_type == 'bernoulli':
        assert torch.all((p >= 0) & (p <= 1)), "Probabilities in p must be in the range (0, 1)"
        assert torch.all((q >= 0) & (q <= 1)), "Probabilities in q must be in the range (0, 1)"

        m = 0.5 * (p + q)
        jsd_value = 0.5 * (binary_entropy(m) - 0.5 * binary_entropy(p) - 0.5 * binary_entropy(q))

    elif distribution_type == 'multinomial':
        assert torch.all(p >= 0), "Probabilities in p must be non-negative"
        assert torch.all(q >= 0), "Probabilities in q must be non-negative"

        assert torch.allclose(torch.sum(p, dim=-1), torch.tensor(1.0), atol=epsilon), \
            "Sum of probabilities in p is not approximately one"

        assert torch.allclose(torch.sum(q, dim=-1), torch.tensor(1.0), atol=epsilon), \
            "Sum of probabilities in q is not approximately one"

        m = 0.5 * (p + q)

        m = torch.maximum(torch.full_like(m, epsilon), m)

        kl_pm = torch.sum(p * torch.log(p / m), dim=-1)

        kl_qm = torch.sum(q * torch.log(q / m), dim=-1)

        jsd_value = 0.5 * (kl_pm + kl_qm)

    elif distribution_type == 'gaussian':
        jsd_value = torch.linalg.norm(p - q, dim=-1) / 16

    else:
        raise ValueError("Invalid distribution type. Possible values: 'bernoulli', 'multinomial', 'gaussian'.")

    return jsd_value


def model_jsd(model_1, model_2, dataloader, task_type, device, epsilon=1e-10):
    """
    Calculate Jensen-Shannon Divergence (JSD) between the output distributions of
    the constructed model and a reference model.

    Parameters:
    - model_1 (torch.nn.Module): model for comparison.
    - model_2 (torch.nn.Module): model for comparison.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing input data for both models.
    - task_type (str): Type of the task, one of "binary_classification", "classification", or "regression".
    - epsilon (float): A small value added to the probabilities to avoid division by zero, default is 1e-10.

    Returns:
    - jsd_value (float): Jensen-Shannon Divergence between the output distributions of the two models.
    """
    model_1.eval()
    model_2.eval()

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    outputs_list_1 = []
    outputs_list_2 = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device).type(torch.float32)
            outputs_list_1.append(model_1(inputs))
            outputs_list_2.append(model_2(inputs))

    outputs_1 = torch.cat(outputs_list_1)
    outputs_2 = torch.cat(outputs_list_2)

    if task_type == "binary_classification":
        outputs_1 = torch.sigmoid(outputs_1)
        outputs_2 = torch.sigmoid(outputs_2)
        distribution_type = 'bernoulli'

    elif task_type == "classification":
        softmax = nn.Softmax(dim=1)

        outputs_1 = softmax(outputs_1)
        outputs_2 = softmax(outputs_2)
        distribution_type = 'multinomial'

    elif task_type == "regression":
        distribution_type = 'gaussian'

    else:
        raise NotImplementedError(
            f"Task {task_type} is not supported."
            "Possible are: 'binary_classification', 'classification', 'regression'."
        )

    score = jsd(outputs_1, outputs_2, distribution_type=distribution_type, epsilon=epsilon).mean(axis=0)

    return float(score)


def binary_entropy(p):
    """
    Calculate the binary entropy for a given probability.

    The binary entropy for a probability distribution with two outcomes is defined as:
        H(p) = -p * log2(p) - (1 - p) * log2(1 - p)

    Parameters:
    - p (torch.Tensor): A tensor containing probabilities in the range [0, 1].

    Returns:
    torch.Tensor: The binary entropy of the input probabilities.
    """
    assert torch.all((p >= 0) & (p <= 1)), "Probabilities in p must be in the range (0, 1)"

    return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)


def gumbel_softmax(
        logits, tau: float = 1., hard: bool = False, binary: bool = False, threshold: float = 0.5,
        generator=None, dim: int = -1
):
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
        logits: `[..., num_features]` unnormalized log probabilities
        tau: non-negative scalar temperature
        hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
        binary: if ``True``,
        threshold:
        generator (torch.Generator, optional): a pseudorandom number generator for sampling
        dim: A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using re-parametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, dim=dim)

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=generator).log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)

    if binary:
        y_soft = gumbels.sigmoid()
    else:
        y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        if binary:
            y_hard = (y_soft > threshold).type(torch.int64)
        else:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)

        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Re-parametrization trick.
        ret = y_soft
    return ret


def get_most_probable_class(logits, threshold=None, dim=-1):
    """
    Get the most probable class from logits.

    Args:
        logits (torch.Tensor): The logits for each class. For binary classification, the tensor
                              should have shape (batch_size, 1). For categorical
                              classification, the tensor should have shape (batch_size, num_classes).
        threshold (float, optional): Threshold for binary classification. If None, the default is 0.0.
        dim (int, optional): The dimension along which the argmax operation is performed. Default is -1.

    Returns:
        torch.Tensor: A tensor containing the index of the most probable class for each input example
                      in the batch.

    Example:
        >>> binary_logits_batch = torch.tensor([[0.7], [-0.3], [0.8]])
        >>> categorical_logits_batch = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
        >>> most_probable_class_binary_batch = get_most_probable_class(binary_logits_batch)
        >>> most_probable_class_categorical_batch = get_most_probable_class(categorical_logits_batch)
        >>> print("Most probable class (binary):", most_probable_class_binary_batch)
        >>> print("Most probable class (categorical):", most_probable_class_categorical_batch)
    """
    if logits.size(-1) == 1:
        if threshold is None:
            threshold = 0.0
        return (logits > threshold).float()
    else:
        _, predicted_classes = torch.max(logits, dim)
        return predicted_classes.float()
