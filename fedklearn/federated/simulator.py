import os
import logging
from abc import ABC, abstractmethod

import torch
import numpy as np
from copy import copy

from .utils import *
import torch.optim as optim
from ..trainer.trainer import Trainer

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
    - use_dp (bool): Flag to determine whether to use differential privacy.
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

    def __init__(self, clients, global_trainer, logger, chkpts_dir, use_dp=False, rng=None):
        """
         Initialize the federated learning simulator.

         Parameters:
         - clients (list): List of clients participating in the simulation.
         - global_trainer: The global trainer responsible for model aggregation.
         - logger: The logger for recording simulation logs.
         - chkpts_dir (str): Directory to save simulation checkpoints.
         - log_freq (int): Frequency for logging simulation information.
         - rng: Random number generator for reproducibility.
         - use_dp (bool): Flag to determine whether to use differential privacy.
         """

        self.clients = clients

        self.global_trainer = global_trainer

        self.logger = logger

        self.chkpts_dir = chkpts_dir

        self.use_dp = use_dp

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
            epsilon_list = []
            if self.use_dp:
                train_loss, train_metric, test_loss, test_metric, epsilon = client.write_logs()
                epsilon_list.append(epsilon)
            else:
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
        if self.use_dp:
            logging.info(f"Avg. Epsilon: {np.mean(epsilon_list):.4f}")
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


class ActiveAdamFederatedAveraging(FederatedAveraging):
    """
    ActiveAdamFederatedAveraging class extends the FederatedAveraging class to simulate an active server that modify the
    clients' weights to accelerate clients' convergence.


    """

    def __init__(self, clients, global_trainer, logger, chkpts_dir, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 alpha=0.1, attacked_round=99, active_chkpts_dir=None, use_dp=False, rng=None):
        """
        Initialize the federated learning simulator.

        Parameters:
        - clients (list): List of clients participating in the simulation.
        - global_trainer: The global trainer responsible for model aggregation.
        - logger: The logger for recording simulation logs.
        - chkpts_dir (str): Directory to save simulation checkpoints.
        - log_freq (int): Frequency for logging simulation information.
        - rng: Random number generator for reproducibility.
        - beta1 (float): The exponential decay rate for the first moment estimates.
        - beta2 (float): The exponential decay rate for the second moment estimates.
        - epsilon (float): A small constant for numerical stability.
        - alpha (float): The learning rate for the server.
        - attacked_round (int): The round in which the active simulation starts.
        - pseudo_gradients (list): List of each client current pseudo-gradient value.
        - active_chkpts_dir (str, opt): Directory to save simulation checkpoints for the active server.

        Methods:
        - _initialize_server_trainers: Initialize server models to actively update clients' weights.
        - simulate_server_updates: Update the server's copies of the clients' weights.
        - compute_pseudo_gradient: Compute the pseudo-gradient for the client model based on the server model.
        - active_update: Update client's weights to accelerate clients' convergence.
        - simulate_active_round: Simulate one round of federated learning using an active malicious server.
        - write_logs: Write simulation logs using the logger.
        - save_checkpoints: Save simulation checkpoints to the specified directory.
        - simulate_round: Simulate one round of federated learning using Federated Averaging.
        """

        super().__init__(clients, global_trainer, logger, chkpts_dir, use_dp, rng)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
        self.server_trainers = self._init_server_trainers()
        self.attacked_round = attacked_round
        self.active_chkpts_folders_dict = self._create_active_chkpts_folders(active_chkpts_dir)
        self.messages_metadata = self._init_messages_metadata()
        self.pseudo_gradients = self._init_pseudo_gradients()


    def _init_pseudo_gradients(self):
        pseudo_gradients = []
        for i in range(len(self.clients)):
            pseudo_gradients.append(torch.zeros_like(self.clients[i].trainer.get_param_tensor()))

        return pseudo_gradients

    def _create_active_chkpts_folders(self, active_chkpts_folder=None):
        """
                Create checkpoint folders for the global model and each client.

                This method ensures the existence of the checkpoint directory and creates
                individual folders for each client within the checkpoint directory based on the attacked round.

                Note: This method does not modify the clients but creates the necessary directory structure.

                Returns:
                    - dict: A dictionary containing paths to the checkpoint folders for the global model
                        and each client, with client names as keys.
                """

        chkpts_folders_dict = dict()

        os.makedirs(self.chkpts_dir, exist_ok=True)
        if active_chkpts_folder is None:
            server_model_folder = os.path.join(self.chkpts_dir, "active_server")
        else:
            server_model_folder = os.path.join(active_chkpts_folder, "active_server")
        os.makedirs(server_model_folder, exist_ok=True)
        chkpts_folders_dict["server"] = dict()

        for client in self.clients:
            if active_chkpts_folder is None:
                active_server_path = os.path.join(self.chkpts_dir, 'active_server', client.name, f'{self.attacked_round}')
                path = os.path.join(self.chkpts_dir, client.name, 'active', f'{self.attacked_round}')
            else:
                active_server_path = os.path.join(active_chkpts_folder, 'active_server', client.name, f'{self.attacked_round}')
                path = os.path.join(active_chkpts_folder, client.name, f'{self.attacked_round}')

            os.makedirs(path, exist_ok=True)
            os.makedirs(active_server_path, exist_ok=True)

            chkpts_folders_dict[client.name] = path
            chkpts_folders_dict['server'][client.name] = active_server_path

        return chkpts_folders_dict


    def _init_messages_metadata(self):
        """
        Initialize metadata for messages exchanged between the global server and clients.

        This method creates and returns a dictionary (`messages_metadata`) to store metadata
        related to messages exchanged during federated learning. The dictionary includes an
        entry for the global server, for each client and for each active server exchange,
        with initial empty metadata.

        Returns:
        - dict: A dictionary containing metadata for messages, with keys for the global server, the active server,
                and each client, initialized with empty metadata dictionaries.
        """

        messages_metadata = {"server": dict()}
        messages_metadata['global'] = dict()
        for client in self.clients:
            messages_metadata[client.name] = dict()
            messages_metadata['server'][client.name] = dict()

        return messages_metadata

    def _init_gradients(self, trainer):
        """
        Initialize the gradients of the pseudo-gradients.

        Args:
        - trainer: Trainer of the client.

        Returns:
        - Trainer: Trainer with gradients initialized to zero.

        """
        for param in trainer.model.parameters():
            param.grad = torch.zeros_like(param)

        return trainer


    def _init_server_trainers(self):
        """
        Initialize server models to actively update clients' weights.
        Returns:
        - list: List of  trainers.

        """
        server_trainers = []
        for i in range(len(self.clients)):
            # TODO: maybe can be cleaned. Cannot deepcopy the trainer due to hooks erors with opacus

            model = copy.deepcopy(self.clients[i].trainer.model)
            criterion = copy.deepcopy(self.clients[i].trainer.criterion)
            lr_scheduler = copy.deepcopy(self.clients[i].trainer.lr_scheduler)
            is_binary_classification = copy.deepcopy(self.clients[i].trainer.is_binary_classification)
            device = copy.deepcopy(self.clients[i].trainer.device)
            model_name = copy.deepcopy(self.clients[i].trainer.model_name)
            metric = copy.deepcopy(self.clients[i].trainer.metric)
            optimizer = optim.Adam(model.parameters(),
                                                  lr=self.alpha,
                                                  betas=(self.beta1, self.beta2)
                                                  )
            server_trainer = Trainer(
                model=model,
                criterion=criterion,
                metric=metric,
                device=device,
                model_name=model_name,
                lr_scheduler=lr_scheduler,
                is_binary_classification=is_binary_classification,
                optimizer=optimizer,
            )
            # server_trainer = copy.deepcopy(self.clients[i].trainer)
            server_trainers.append(server_trainer)
        return server_trainers


    def compute_pseudo_gradient(self, client_trainer, server_trainer):
        """
        Compute the pseudo-gradient for the client model based on the server model.

        Args:
            client_trainer: Trainer of the attacked client.
            server_trainer: Trainer of the server copy of the client.

        Returns:
            - torch.Tensor: The pseudo-gradient for the client model.
        """
        return client_trainer.get_param_tensor() - server_trainer.get_param_tensor()


    def active_update(self, task_index):
        """
        Update client's parameters to accelerate clients' convergence, performing the following steps:
        1. Compute the pseudo-gradient for each client.
        2. Compute an Adam update for each client copy.
        3. Update the client's parameters using the ones received from the server.

        Args:
            task_index(int): Index of the client in the clients list.

        """
        self.server_trainers[task_index] = self._init_gradients(self.server_trainers[task_index])
        pseudo_gradient = self.compute_pseudo_gradient(self.clients[task_index].trainer, self.server_trainers[task_index])

        self.server_trainers[task_index].set_grad_tensor(-pseudo_gradient)
        self.pseudo_gradients[task_index] = pseudo_gradient

        self.server_trainers[task_index].optimizer.step()
        self.server_trainers[task_index].optimizer.zero_grad()

        # self.clients[task_index].trainer.set_param_tensor(self.server_trainers[task_index].get_param_tensor())
        self.clients[task_index].update_trainer(self.server_trainers[task_index])

    def simulate_server_updates(self):
        """
        Simulate server updates on each client.

        This method iterates through each client and simulates a server update.
        """
        for i, client in enumerate(self.clients):
            self.active_update(i)


    def simulate_active_round(self, save_chkpts=False, save_logs=False):
        """
        Simulate one round of federated learning using an active malicious server.

        Parameters:
        - save_chkpts (bool): Flag to determine whether to save checkpoints.
        - save_logs (bool): Flag to determine whether to save simulation logs.

        """

        logging.debug(f"Round {self.c_round}:")

        self.simulate_local_updates()

        if save_chkpts:
            self.save_active_checkpoints(mode='client')

        self.simulate_server_updates()
        logging.debug(f"Server update computed successfully")

        if save_chkpts:
            self.save_active_checkpoints(mode='server')
            logging.debug(
                f"Checkpoint saved and messages metadata updated successfully at communication round {self.c_round}."
            )

        if save_logs:
            self.write_logs()

        self.c_round += 1

    def write_logs(self, display_only=True):
        """
        Write simulation logs using the logger.
        """
        global_train_loss = 0.
        global_train_metric = 0.
        global_test_loss = 0.
        global_test_metric = 0.

        total_n_samples = 0
        total_n_test_samples = 0

        logging.info("+" * 50)
        for client_id, client in enumerate(self.clients):
            if self.use_dp:
                train_loss, train_metric, test_loss, test_metric, epsilon = client.write_logs()
                logging.info(f"Client {client.name} | Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} | Epsilon Spent: {epsilon:.4f}")
            else:
                train_loss, train_metric, test_loss, test_metric = client.write_logs()
                logging.info(f"Client {client.name} | Train Loss: {train_loss:.4f} | Train Metric: {train_metric:.4f} |")

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

        if not display_only:
            self.logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            self.logger.add_scalar("Train/Metric", global_train_metric, self.c_round)
            self.logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            self.logger.add_scalar("Test/Metric", global_test_metric, self.c_round)

        return global_train_loss, global_train_metric, global_test_loss, global_test_metric


    def save_active_checkpoints(self, mode='client'):
        """
        Save simulation checkpoints to the specified directory.
        Args:
            mode (str): Indicates whether to save the checkpoints for the clients or the server.

        Returns:

        """
        if mode == 'client':
            for client in self.clients:
                path = os.path.join(self.active_chkpts_folders_dict[client.name], f"{self.c_round}.pt")
                path = os.path.abspath(path)
                client.trainer.save_checkpoint(path)

                self.messages_metadata[client.name][self.c_round] = path

        elif mode == 'server':
            for i, client in enumerate(self.clients):
                path = os.path.join(self.active_chkpts_folders_dict['server'][client.name], f"{self.c_round}.pt")
                path = os.path.abspath(path)
                client.trainer.save_checkpoint(path)

                self.messages_metadata['server'][client.name][self.c_round] = path

        else:
            raise ValueError(f"Invalid mode: {mode}. Valid options are 'client' and 'server'.")


    def get_client_avg_train_loss(self):
        """
        Get the average train loss of the clients.

        Returns:
        - float: The average train loss of the clients.
        """
        total_train_loss = 0
        total_n_samples = 0
        for client in self.clients:
            client_loss, _ = client.evaluate_train_loader()
            total_train_loss += client_loss * client.n_train_samples
            total_n_samples += client.n_train_samples
        return total_train_loss / total_n_samples

    def compute_pseudo_grad_norm(self):
        """
        Get the average norm of the pseudo-gradients of the clients.
        Returns:
        - float: The average norm of the pseudo-gradients of the clients.

        """
        norm_list = [torch.norm(pseudo_grad) for pseudo_grad in self.pseudo_gradients]
        n_samples = [client.n_train_samples for client in self.clients]

        weighted_sum = sum(score * n_sample for score, n_sample in zip(norm_list, n_samples))
        total_samples = sum(n_samples)
        return weighted_sum / total_samples
