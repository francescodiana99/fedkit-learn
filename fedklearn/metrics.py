import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Mean Squared Error.
    """
    return F.mse_loss(y_pred, y_true, reduction='mean')


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Mean Absolute Error.
    """
    return F.l1_loss(y_pred, y_true, reduction='mean')


def r2_score(y_true, y_pred):
    """
    Calculate the R-squared (R2) Score between true and predicted values.

    Parameters:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: R-squared (R2) Score.
    """
    mean_y_true = torch.mean(y_true)
    total_variance = torch.sum((y_true - mean_y_true)**2)
    residual_variance = torch.sum((y_true - y_pred)**2)
    r2 = 1 - (residual_variance / total_variance)

    return r2


def multiclass_accuracy(y_pred, y_true):
    """
    Calculate multiclass accuracy.

    Parameters:
    - y_pred (torch.Tensor): Tensor containing predicted values.
    - y_true (torch.Tensor): Tensor containing true labels.

    Returns:
    - float: Multiclass accuracy.
    """
    assert y_pred.shape[0] == y_true.shape[0], "Shapes of predictions and targets must match."

    _, predicted_labels = torch.max(y_pred, dim=1)

    correct_predictions = (predicted_labels == y_true).float()

    accuracy = correct_predictions.sum() / len(y_true)

    return accuracy.item()


def binary_accuracy_with_sigmoid(y_pred, y_true):
    """
    Calculate binary accuracy. Applies sigmoid activation to y_pred before rounding.

    Parameters:
    - y_pred (torch.Tensor): Tensor containing predicted values (0 or 1).
    - y_true (torch.Tensor): Tensor containing true labels (0 or 1).

    Returns:
    - float: Binary accuracy.
    """
    assert y_pred.shape == y_true.shape, "Shapes of predictions and targets must match."

    predicted_labels = torch.round(torch.sigmoid(y_pred))

    correct_predictions = (predicted_labels == y_true).float()

    accuracy = correct_predictions.sum() / len(y_true)

    return accuracy.item()


def threshold_binary_accuracy(y_pred, y_true, threshold=1e-12):
    """
    Calculate accuracy based on the difference between elements and a given threshold.

    Parameters:
        y_pred (torch.Tensor): Tensor containing predicted values (e.g., model outputs).
        y_true (torch.Tensor): Tensor containing target values (ground truth labels).
        threshold (float): Threshold for considering a prediction as correct.

    Returns:
        float: Accuracy.
    """
    differences = torch.abs(y_pred - y_true)

    correct_predictions = (differences <= threshold).float()

    # Calculate accuracy
    accuracy = correct_predictions.mean().item()

    return accuracy


def binary_accuracy(y_pred, y_true):
    """
    Calculate binary accuracy. No rounding is performed

    Parameters:
    - y_pred (torch.Tensor): Tensor containing predicted values.
    - y_true (torch.Tensor): Tensor containing true labels.

    Returns:
    - float: Binary accuracy.
    """
    assert y_pred.shape == y_true.shape, "Shapes of predictions and targets must match."

    predicted_labels = torch.round(y_pred)

    correct_predictions = (predicted_labels == y_true).float()

    accuracy = correct_predictions.sum() / len(y_true)

    return accuracy.item()


class CosineDiSimilarityLoss(nn.Module):
    """
    Calculates the cosine dissimilarity loss between input vectors and target vectors.

    The cosine dissimilarity loss is computed as 1.0 minus the cosine similarity between
    the input vectors and target vectors. It penalizes the model if the cosine similarity
    is close to 1 (indicating similarity), encouraging dissimilarity.

    Parameters:
        dim (int, optional): Dimension along which cosine similarity is computed.
            Default is 1, which assumes the input vectors are row vectors.
        eps (float, optional): Small value to prevent division by zero in the cosine similarity
            computation. Default is 1e-08.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - Input: (N, D) where N is the batch size and D is the dimension of the input vectors.
        - Target: (N, D) with the same shape as the input.

    Example:
        >>> criterion = CosineDiSimilarityLoss()
        >>> input_vector = torch.randn(3, 5, requires_grad=True)
        >>> target_vector = torch.randn(3, 5, requires_grad=False)
        >>> loss = criterion(input_vector, target_vector)
    """
    def __init__(self, dim=1, eps=1e-08, reduction: str = 'mean'):
        super(CosineDiSimilarityLoss, self).__init__()
        self.dim = dim
        self.eps = eps

        self.reduction = reduction

    def forward(self, input_, target):
        res = 0.5 * (1. - F.cosine_similarity(input_, target, self.dim, self.eps))

        if self.reduction == "mean":
            return res.mean()

        elif self.reduction == "sum":
            return res.sum()

        elif self.reduction == "none":
            return res

        else:
            raise RuntimeError(
                f"{self.reduction} is not a valid reduction. Possible are: ``'none'`` | ``'mean'`` | ``'sum'``"
            )


class CosineMSELoss(nn.Module):
    """
    Combined loss function that is the sum of cosine dissimilarity and Mean Squared Error (MSE) loss.

    Parameters:
        mse_weight (float, optional): Weight for the MSE loss term in the total loss.
            Default is 1.0.
        dim (int, optional): Dimension along which cosine similarity is computed.
            Default is 1, which assumes the input vectors are row vectors.
        eps (float, optional): Small value to prevent division by zero in the cosine similarity
            computation. Default is 1e-08.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - Input: (N, D) where N is the batch size and D is the dimension of the input vectors.
        - Target: (N, D) with the same shape as the input.

    Example:
        >>> criterion = CosineMSELoss(cosine_margin=0.1, mse_weight=1.0)
        >>> input_vector = torch.randn(3, 5, requires_grad=True)
        >>> target_vector = torch.randn(3, 5, requires_grad=False)
        >>> loss = criterion(input_vector, target_vector)
    """
    def __init__(self, dim=1, eps=1e-08, mse_weight=1.0, reduction: str = 'mean'):
        super(CosineMSELoss, self).__init__()

        self.mse_weight = mse_weight

        self.cosine_loss = CosineDiSimilarityLoss(dim=dim, eps=eps, reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input_, target):
        cosine_loss = self.cosine_loss(input_, target)

        mse_loss = self.mse_loss(input_, target)

        total_loss = cosine_loss + self.mse_weight * mse_loss

        return total_loss
