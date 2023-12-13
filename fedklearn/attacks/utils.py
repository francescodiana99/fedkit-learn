from torch.utils.data._utils.collate import default_collate


def get_all_features(dataset):
    """
    Retrieve all features and labels from the federated learning dataset.

    Parameters:
        dataset (torch.utils.data.Dataset)

    Returns:
    - Tuple of torch.Tensors: A tuple containing tensors representing all features and labels in the dataset.
    """
    all_examples = [example for example in dataset]

    return default_collate(all_examples)
