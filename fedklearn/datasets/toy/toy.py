import os
import json
import logging

import numpy as np

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
    - sensitive_attribute_type (str): Type of the sensitive feature, either 'numerical' or 'binary'.
    - sensitive_attribute_weight (float): Weight of the sensitive feature.
    - bias (bool): If `True`, a bias term is included in the linear model.
    - noise_level (float): Optional noise level to add to labels.
    - cache_dir (str): Directory to store generated datasets.
    - force_generation (bool): If True, forces regeneration of datasets even if cached ones exist.
    - allow_generation (bool): If False, the dataset is not generated.
        If the data is not previously cached and allow_generation is False,
        the class initialization will throw a runtime error.
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
    - ValueError: If an invalid problem type or sensitive feature type is specified.
    - RuntimeError: If allow_generation is False and no cached data is found.
    """
    def __init__(
            self, cache_dir="./", n_tasks=None, n_train_samples=None, n_test_samples=None, problem_type=None,
            n_numerical_features=None, n_binary_features=None, sensitive_attribute_type=None,
            sensitive_attribute_weight=None, bias=False, noise_level=None, force_generation=False,
            allow_generation=True, rng=None
    ):

        if any(param is None for param in [n_tasks, n_train_samples, n_test_samples, problem_type,
                                           n_numerical_features, n_binary_features, sensitive_attribute_type,
                                           sensitive_attribute_weight, noise_level]):
            allow_generation = False
            logging.warning(f"force_generation is automatically set to False.")

            self.rng = rng if rng is not None else np.random.default_rng()

        self.cache_dir = cache_dir
        self.allow_generation = allow_generation
        self.force_generation = force_generation if self.allow_generation else False

        self.tasks_dir = os.path.join(self.cache_dir, 'tasks')
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')

        if self.force_generation != force_generation:
            logging.warning(f"force_generation is automatically set to {self.force_generation}")

        if not self.allow_generation and not os.path.exists(self.metadata_path):
            raise RuntimeError("Data generation is not allowed, and no cached data is found.")

        if os.path.exists(self.metadata_path) and not self.force_generation:
            logging.info("Processed data folders found in the tasks directory. Loading existing files.")
            self._load_metadata()

            self.task_id_to_name = {i: f"{i}" for i in range(self.n_tasks)}

        else:
            assert problem_type in ['classification', 'regression'], \
                "Invalid problem type. Use 'classification' or 'regression'."

            assert sensitive_attribute_type in ['numerical', 'binary'], \
                "Invalid sensitive feature type. Use 'numerical' or 'binary'."

            if sensitive_attribute_type == 'binary':
                assert n_binary_features > 0, \
                    "If the sensitive feature is binary, the number of binary features should be greater than 0."

            if sensitive_attribute_type == 'numerical':
                assert n_numerical_features > 0, \
                    "If the sensitive feature is numerical, the number of numerical features should be greater than 0."

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

            self.sensitive_attribute_type = sensitive_attribute_type
            self.sensitive_attribute_weight = sensitive_attribute_weight

            self.noise_level = noise_level

            self.bias = bias

            self.rng = rng if rng is not None else np.random.default_rng()

            self.task_id_to_name = {i: f"{i}" for i in range(self.n_tasks)}

            self.sensitive_attribute_id = self._get_sensitive_attribute_id()

            self.weights, self.bias = self._initialize_model_parameters()

            logging.info("==> Generating data..")
            os.makedirs(self.tasks_dir, exist_ok=True)

            for task_id in tqdm(range(self.n_tasks), leave=False):
                train_features, train_labels, test_features, test_labels = self.generate_task_data(task_id=task_id)

                task_dir = os.path.join(self.tasks_dir, f"{task_id}")

                train_save_path = os.path.join(task_dir, "train.npz")
                test_save_path = os.path.join(task_dir, "test.npz")

                os.makedirs(task_dir, exist_ok=True)
                np.savez_compressed(train_save_path, features=train_features, labels=train_labels)
                np.savez_compressed(test_save_path, features=test_features, labels=test_labels)

            self._save_metadata()
            logging.info("data and metadata generated and saved successfully.")

    def _get_sensitive_attribute_id(self):
        if self.sensitive_attribute_type == "numerical":
            sensitive_attribute_id = self.rng.integers(low=0, high=self.n_numerical_features)
        elif self.sensitive_attribute_type == "binary":
            sensitive_attribute_id = self.rng.integers(low=self.n_numerical_features, high=self.n_features + 1)
        else:
            raise ValueError(
                f"Invalid sensitive feature type `{self.sensitive_attribute_type}`. Use 'numerical' or 'binary'."
            )

        return sensitive_attribute_id

    def _initialize_model_parameters(self):
        weights = self.rng.standard_normal(size=(self.n_numerical_features + self.n_binary_features, 1))
        if self.bias:
            bias = self.rng.standard_normal(size=1)
        else:
            bias = np.zeros(shape=1)

        modified_weights = weights.copy()

        # Modify the weights at the specified index
        modified_weights[self.sensitive_attribute_id] = (
            np.sign(weights[self.sensitive_attribute_id]) * np.sqrt(self.sensitive_attribute_weight)
        )

        # Normalize and modify the weights at the complement index
        complement_idx = ~self.sensitive_attribute_id
        norm_factor = np.linalg.norm(weights[complement_idx])
        modified_weights[complement_idx] = (
            (weights[complement_idx] / norm_factor) * np.sqrt(1 - self.sensitive_attribute_weight)
        )

        modified_weights /= np.linalg.norm(modified_weights)

        return modified_weights, bias

    def _save_metadata(self):
        metadata = {
            'n_tasks': self.n_tasks,
            'n_train_samples': self.n_train_samples,
            'n_test_samples': self.n_test_samples,
            'problem_type': self.problem_type,
            'n_numerical_features': self.n_numerical_features,
            'n_binary_features': self.n_binary_features,
            'n_features': self.n_features,
            'sensitive_attribute_type': self.sensitive_attribute_type,
            'sensitive_attribute_weight': self.sensitive_attribute_weight,
            'noise_level': self.noise_level,
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'sensitive_attribute_id': int(self.sensitive_attribute_id),
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

    def _load_metadata(self):
        with open(self.metadata_path, 'r') as json_file:
            metadata = json.load(json_file)

        self.n_tasks = metadata['n_tasks']
        self.n_train_samples = metadata['n_train_samples']
        self.n_test_samples = metadata['n_test_samples']
        self.problem_type = metadata['problem_type']
        self.n_numerical_features = metadata['n_numerical_features']
        self.n_binary_features = metadata['n_binary_features']
        self.n_features = metadata['n_features']
        self.sensitive_attribute_type = metadata['sensitive_attribute_type']
        self.sensitive_attribute_weight = metadata['sensitive_attribute_weight']
        self.noise_level = metadata['noise_level']
        self.weights = np.array(metadata['weights'])
        self.bias = np.array(metadata['bias'])
        self.sensitive_attribute_id = metadata['sensitive_attribute_id']

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
        binary_data = 2. * binary_data - 1.

        features = np.concatenate((numerical_data, binary_data.astype(float)), axis=1)

        logits = np.dot(features, self.weights) + self.bias

        if task_id >= self.n_tasks // 2:
            logits = -logits

        logits += self.noise_level * np.random.standard_normal(size=logits.shape)

        if self.problem_type == 'classification':
            labels = (logits > 0).astype(np.int64).squeeze()
        elif self.problem_type == 'regression':
            labels = logits.squeeze().astype(np.float32)
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

        dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
        dataset.name = f"{task_id}"

        return dataset
