import torch
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
    return F.mse_loss(y_pred, y_true)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Mean Absolute Error.
    """
    return F.l1_loss(y_pred, y_true)


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


def threshold_binary_accuracy(predictions, targets, threshold=1e-12):
    """
    Calculate accuracy based on the difference between elements and a given threshold.

    Parameters:
        predictions (torch.Tensor): Tensor containing predicted values (e.g., model outputs).
        targets (torch.Tensor): Tensor containing target values (ground truth labels).
        threshold (float): Threshold for considering a prediction as correct.

    Returns:
        float: Accuracy.
    """
    differences = torch.abs(predictions - targets)

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
