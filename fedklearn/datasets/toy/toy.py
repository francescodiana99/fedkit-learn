import os
import json
import logging

import numpy as np
from scipy.special import expit

import torch
from torch.utils.data import TensorDataset

from tqdm import tqdm


class FederatedToyDataset:
    """
    A class for generating synthetic federated datasets for classification or regression tasks.

    Parameters:
    - n_tasks (int): Number of tasks in the federated dataset.
    - n_train_samples (int): Number of samples in the training set for each task.
    - n_test_samples (int): Number of samples in the testing set for each task.
    - problem_type (str): Type of the problem, either 'classification' or 'regression'.
    - n_numerical_features (int): Number of numerical features in the dataset.
    - n_binary_features (int): Number of binary features in the dataset.
    - important_feature_type (str): Type of the important feature, either 'numerical' or 'binary'.
    - important_feature_weight (float): Weight of the important feature.
    - noise_level (float): Optional noise level to add to labels.
    - cache_dir (str): Directory to store generated datasets.
    - force_generation (bool): If True, forces regeneration of datasets even if cached ones exist.
    - rng (numpy.random.Generator): Random number generator for reproducibility.

    Methods:
    - generate_task_data(task_id):
        Generates synthetic data for a specific task.

    - get_task_dataset(task_id, mode='train'):
        Retrieves a TensorDataset for a specific task and mode ('train' or 'test').

    Notes:
    - The generated dataset includes numerical and binary features with optional noise.
    - For classification tasks, the class applies a sigmoid function to logits and introduces noise.
    - For regression tasks, the logits are used directly as labels with added noise.
    - The class supports both classification and regression tasks based on the specified problem type.

    Raises:
    - ValueError: If an invalid problem type or important feature type is specified.
    """
    def __init__(
            self, n_tasks, n_train_samples, n_test_samples, problem_type, n_numerical_features, n_binary_features,
            important_feature_type, important_feature_weight, noise_level=None, cache_dir="./", force_generation=False,
            rng=None

    ):
        assert problem_type in ['classification', 'regression'], \
            "Invalid problem type. Use 'classification' or 'regression'."

        assert important_feature_type in ['numerical', 'binary'], \
            "Invalid important feature type. Use 'numerical' or 'binary'."

        if important_feature_type == 'binary':
            assert n_binary_features > 0, \
                "If the important feature is binary, the number of binary features should be greater than 0."

        if important_feature_type == 'numerical':
            assert n_numerical_features > 0, \
                "If the important feature is numerical, the number of numerical features should be greater than 0."

        assert (n_numerical_features + n_binary_features) > 0, \
            "The total number of features should be greater than 0."

        self.n_tasks = n_tasks

        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.n_samples = self.n_train_samples + self.n_test_samples

        self.problem_type = problem_type

        self.n_numerical_features = n_numerical_features
        self.n_binary_features = n_binary_features
        self.n_features = self.n_binary_features + self.n_numerical_features

        self.important_feature_type = important_feature_type
        self.important_feature_weight = important_feature_weight

        self.noise_level = noise_level

        self.cache_dir = cache_dir

        self.force_generation = force_generation

        self.rng = rng if rng is not None else np.random.default_rng()

        self.tasks_dir = os.path.join(self.cache_dir, 'tasks')
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')

        if os.path.exists(self.metadata_path) and not self.force_generation:
            logging.info("Processed data folders found in the tasks directory. Loading existing files.")

            self._load_metadata()

        else:
            os.makedirs(self.tasks_dir, exist_ok=True)

            self.important_feature_idx = self._get_important_feature_idx()

            self.weights, self.bias = self._initialize_model_parameters()

            logging.info("==> Generating data..")
            for task_id in tqdm(range(self.n_tasks), leave=False):
                train_features, train_labels, test_features, test_labels = self.generate_task_data(task_id=task_id)

                task_dir = os.path.join(self.tasks_dir, f"{task_id}")

                train_save_path = os.path.join(task_dir , "train.npz")
                test_save_path = os.path.join(task_dir, "test.npz")

                os.makedirs(task_dir, exist_ok=True)
                np.savez_compressed(train_save_path, features=train_features, labels=train_labels)
                np.savez_compressed(test_save_path, features=test_features, labels=test_labels)

            self._save_metadata()

    def _get_important_feature_idx(self):
        if self.important_feature_type == "numerical":
            important_feature_idx = self.rng.integers(low=0, high=self.n_numerical_features)
        elif self.important_feature_type == "binary":
            important_feature_idx = self.rng.integers(low=self.n_numerical_features, high=self.n_features + 1)
        else:
            raise ValueError(
                f"Invalid important feature type `{self.important_feature_type}`. Use 'numerical' or 'binary'."
            )

        return important_feature_idx

    def _initialize_model_parameters(self):
        weights = self.rng.standard_normal(size=(self.n_numerical_features + self.n_binary_features, 1))
        bias = self.rng.standard_normal(size=1)

        modified_weights = weights.copy()

        # Modify the weights at the specified index
        modified_weights[self.important_feature_idx] = (
            np.sign(weights[self.important_feature_idx]) * np.sqrt(self.important_feature_weight)
        )

        # Normalize and modify the weights at the complement index
        complement_idx = ~self.important_feature_idx
        norm_factor = np.linalg.norm(weights[complement_idx])
        modified_weights[complement_idx] = (
            (weights[complement_idx] / norm_factor) * np.sqrt(1 - self.important_feature_weight)
        )

        return modified_weights, bias

    def _save_metadata(self):
        metadata = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'important_feature_idx': int(self.important_feature_idx)
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

    def _load_metadata(self):
        with open(self.metadata_path, 'r') as json_file:
            metadata = json.load(json_file)

        self.weights = np.array(metadata['weights'])
        self.bias = np.array(metadata['bias'])
        self.important_feature_idx = metadata['important_feature_idx']

    def generate_task_data(self, task_id):
        """
        Generates synthetic data for the specific task.

        Parameters:
        - task_id (int): Identifier for the specific task. Affects the dataset generation.

        Returns:
        - train_features (numpy.ndarray): Array of input features for the training dataset.
        - train_labels (numpy.ndarray): Array of labels corresponding to the training features.
        - test_features (numpy.ndarray): Array of input features for the testing dataset.
        - test_labels (numpy.ndarray): Array of labels corresponding to the testing features.

        Notes:
        - The function supports both classification and regression tasks based on the specified problem type.
        - The `task_id` parameter influences the generation process. If `task_id` is greater than or equal to half
          the number of tasks, logits are negated, potentially reversing class labels for classification tasks.
        - The generated dataset includes numerical and binary features with optional noise.
        - For classification tasks, the function applies a sigmoid function to logits and introduces noise.
        - For regression tasks, the logits are used directly as labels with added noise.

        Raises:
        - ValueError: If an invalid problem type is specified. Use 'classification' or 'regression'.
        """
        numerical_data = self.rng.standard_normal(size=(self.n_samples, self.n_numerical_features))

        binary_data = self.rng.integers(low=0, high=2, size=(self.n_samples, self.n_binary_features))

        features = np.concatenate((numerical_data, binary_data.astype(float)), axis=1)

        logits = np.dot(features, self.weights) + self.bias

        if task_id >= self.n_tasks // 2:
            logits = -logits

        logits += self.noise_level * np.random.standard_normal(size=logits.shape)

        if self.problem_type == 'classification':
            probs = expit(-logits)
            labels = self.rng.binomial(1, probs).squeeze()
        elif self.problem_type == 'regression':
            labels = logits.squeeze()
        else:
            raise ValueError("Invalid problem type. Use 'classification' or 'regression'.")

        train_features, test_features = features[:self.n_train_samples], features[self.n_train_samples:]
        train_labels, test_labels = labels[:self.n_train_samples], labels[self.n_train_samples:]

        return train_features, train_labels, test_features, test_labels

    def get_task_dataset(self, task_id, mode='train'):
        """
        Retrieves a TensorDataset for a specific task and mode ('train' or 'test').

        Parameters:
        - task_id (int): Identifier for the specific task.
        - mode (str): Mode of the dataset, either 'train' or 'test'.

        Returns:
        - dataset (torch.utils.data.TensorDataset): Dataset for the specified task and mode.

        Raises:
        - ValueError: If an invalid mode is specified. Supported values are 'train' or 'test'.
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Supported values are 'train' or 'test'.")

        task_data = np.load(os.path.join(self.cache_dir, 'tasks', f"{task_id}", f'{mode}.npz'))
        features, labels = task_data["features"], task_data["labels"]

        return TensorDataset(torch.tensor(features), torch.tensor(labels))
