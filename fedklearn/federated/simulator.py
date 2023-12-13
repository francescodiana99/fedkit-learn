import os
import logging
from abc import ABC, abstractmethod

import torch
import numpy as np

from .utils import *


class Simulator(ABC):
    """
    Base class for federated learning simulation.

    This class provides a framework for federated learning simulations, allowing users to model
    the interaction between a central server and a set of distributed clients.

    Attributes:
    - clients (list): List of clients participating in the simulation.
    - global_trainer: The global trainer responsible for aggregating models.
    - logger: The logger for recording simulation logs.
    - chkpts_dir (str): Directory to save simulation checkpoints.
    - log_freq (int): Frequency for logging simulation information.
    - message_metadata (dict): Dictionary containing the metadata of the messages exchanged
        between the clients and the server, with client names as keys.
    - chkpts_folders_dict (dict): Dictionary containing paths to the checkpoint folders for the global model
            and each client, with client names as keys.
    - rng: Random number generator for reproducibility.
    - c_round (int): Counter to track the current round.

    Methods:
    - _init_messages_metadata: Initialize metadata for messages exchanged between the global server and clients.
    - _create_chkpts_folders: Create checkpoint folders for the global model and each client.
    - _compute_clients_weights: Compute normalized weights for each client based on
        the number of training samples.
    - _name_clients: Assign default names to clients that are not already named.
    - aggregate: Abstract method for model aggregation.
    - local_updates: Abstract method for simulating local model updates on clients.
    - update_clients: Abstract method for updating clients based on the aggregated model.
    - write_logs: Method for writing simulation logs.
    - save_checkpoints: Method for saving simulation checkpoints.
    - simulate_round: Method for simulating one round of federated learning.
    """

    def __init__(self, clients, global_trainer, logger, chkpts_dir, rng=None):
        """
         Initialize the federated learning simulator.

         Parameters:
         - clients (list): List of clients participating in the simulation.
         - global_trainer: The global trainer responsible for model aggregation.
         - logger: The logger for recording simulation logs.
         - chkpts_dir (str): Directory to save simulation checkpoints.
         - log_freq (int): Frequency for logging simulation information.
         - rng: Random number generator for reproducibility.
         """

        self.clients = clients

        self.global_trainer = global_trainer

        self.logger = logger

        self.chkpts_dir = chkpts_dir

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        self.clients_weights = self._compute_clients_weights()

        self._name_clients()

        self.chkpts_folders_dict = self._create_chkpts_folders()
        logging.info("Checkpoint folders created successfully.")

        self.messages_metadata = self._init_messages_metadata()

        self.c_round = 0

    def _init_messages_metadata(self):
        """
        Initialize metadata for messages exchanged between the global server and clients.

        This method creates and returns a dictionary (`messages_metadata`) to store metadata
        related to messages exchanged during federated learning. The dictionary includes an
        entry for the global server and each client, with initial empty metadata.

        Returns:
        - dict: A dictionary containing metadata for messages, with keys for the global server
                and each client, initialized with empty metadata dictionaries.
        """

        messages_metadata = {"global": dict()}
        for client in self.clients:
            messages_metadata[client.name] = dict()

        return messages_metadata

    def _create_chkpts_folders(self):
        """
        Create checkpoint folders for the global model and each client.

        This method ensures the existence of the checkpoint directory and creates
        individual folders for each client within the checkpoint directory.

        Note: This method does not modify the clients but creates the necessary directory structure.

        Returns:
            - dict: A dictionary containing paths to the checkpoint folders for the global model
                and each client, with client names as keys.
        """
        chkpts_folders_dict = dict()

        os.makedirs(self.chkpts_dir, exist_ok=True)

        global_model_folder = os.path.join(self.chkpts_dir, "global")
        os.makedirs(global_model_folder, exist_ok=True)
        chkpts_folders_dict["global"] = global_model_folder

        for client in self.clients:
            path = os.path.join(self.chkpts_dir, client.name)

            os.makedirs(path, exist_ok=True)

            chkpts_folders_dict[client.name] = path

        return chkpts_folders_dict

    def _compute_clients_weights(self):
        """
        Compute normalized weights for each client based on the number of training samples.

        This method calculates the weights for each client based on the number of training samples.
        The weights are normalized to ensure they sum to 1.

        Note: This method does not modify the clients but returns the computed weights.

        Returns:
        - torch.Tensor: A tensor containing normalized weights for each client based on training samples.
        """
        clients_weights = torch.tensor(
            [client.n_train_samples for client in self.clients],
            dtype=torch.float32
        )

        clients_weights /= clients_weights.sum()

        return clients_weights

    def _name_clients(self):
        """
        Assign default names to clients that are not already named.

        This method checks if each client in the list has a name assigned. If a client
        does not have a name, it assigns a default name of the form "client_i," where
        i is the index of the client in the list.

        Note: This method modifies the clients in-place.
        """
        all_named = all(client.name is not None for client in self.clients)

        if not all_named:
            for client_id, client in enumerate(self.clients):
                if client.name is None:
                    client.name = f"{client_id}"

    @abstractmethod
    def simulate_local_updates(self):
        """
        Abstract method for simulating local model updates on each client.
        """
        pass

    @abstractmethod
    def synchronize(self):
        """
        Abstract method for updating clients based on the aggregated model.
        """
        pass

    @abstractmethod
    def simulate_round(self):
        """
        Abstract method for simulating one round of federated learning.
        """
        pass

    def write_logs(self):
        """
        Write simulation logs using the logger.
        """
        global_train_loss = 0.
        global_train_metric = 0.
        global_test_loss = 0.
        global_test_metric = 0.

        total_n_samples = 0
        total_n_test_samples = 0

        for client_id, client in enumerate(self.clients):

            train_loss, train_metric, test_loss, test_metric = client.write_logs()

            global_train_loss += train_loss * client.n_train_samples
            global_train_metric += train_metric * client.n_train_samples
            global_test_loss += test_loss * client.n_test_samples
            global_test_metric += test_metric * client.n_test_samples

            total_n_samples += client.n_train_samples
            total_n_test_samples += client.n_test_samples

        global_train_loss /= total_n_samples
        global_test_loss /= total_n_test_samples
        global_train_metric /= total_n_samples
        global_test_metric /= total_n_test_samples

        logging.info("+" * 50)
        logging.info(f"Train Loss: {global_train_loss:.4f} | Train Metric: {global_train_metric:.4f} |")
        logging.info(f"Test Loss: {global_test_loss:.4f} | Test Metric: {global_test_metric:.4f} |")
        logging.info("+" * 50)

        self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
        self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
        self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
        self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)

    def save_checkpoints(self):
        """
        Save simulation checkpoints to the specified directory.
        """
        path = os.path.join(self.chkpts_folders_dict["global"], f"{self.c_round}.pt")
        path = os.path.abspath(path)
        self.global_trainer.save_checkpoint(path)

        self.messages_metadata["global"][self.c_round] = path

        for client in self.clients:
            path = os.path.join(self.chkpts_folders_dict[client.name], f"{self.c_round}.pt")
            path = os.path.abspath(path)
            client.trainer.save_checkpoint(path)

            self.messages_metadata[client.name][self.c_round] = path


class FederatedAveraging(Simulator):
    """
    FederatedAveraging class extends the base Simulator class to implement a federated learning simulation
    using the Federated Averaging algorithm.

    Methods:
    - simulate_local_updates: Simulate local model updates on each client.
    - synchronize: Update clients based on the aggregated model.
    - aggregate: Aggregate models from clients to compute a global model using Federated Averaging.
    - simulate_round: Simulate one round of federated learning using Federated Averaging.

    Attributes:
    - Inherits attributes from the base Simulator class.

    """

    def simulate_local_updates(self):
        """
         Simulate local model updates on each client.

         This method iterates through each client and simulates a local model update.

         """
        for client in self.clients:
            client.step()

    def synchronize(self):
        """
        Update clients based on the aggregated model.

        This method iterates through each client and updates its local model using the
        aggregated global model.

        """
        for client in self.clients:
            client.update_trainer(self.global_trainer)

    def aggregate(self):
        """
        Aggregate models from clients to compute a global model using Federated Averaging.

        This method collects models from each client, computes a weighted average, and updates
        the global model.

        """
        models = [client.trainer.model for client in self.clients]
        average_model = weighted_average(models=models, weights=self.clients_weights.tolist())

        self.global_trainer.update_model(model=average_model)

    def simulate_round(self, save_chkpts=False, save_logs=False):
        """
        Simulate one round of federated learning using Federated Averaging.

        Parameters:
        - save_chkpts (bool): Flag to determine whether to save checkpoints.
        - save_logs (bool): Flag to determine whether to save simulation logs.

        """
        logging.debug(f"Round {self.c_round}:")

        self.synchronize()
        logging.debug(f"Clients synchronized successfully")

        self.simulate_local_updates()

        if save_chkpts:
            self.save_checkpoints()
            logging.debug(
                f"Checkpoint saved and messages metadata updated successfully at communication round {self.c_round}."
            )

        self.aggregate()
        logging.debug(f"Global model computed and updated successfully")

        self.synchronize()
        logging.debug(f"Clients synchronized successfully")

        if save_logs:
            self.write_logs()

        self.c_round += 1
