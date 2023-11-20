import logging

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from ..utils import *


class LocalModelReconstructionAttack:
    """
    Class representing a local model reconstruction attack in federated learning.

    This attack aims to approximately reconstruct the local model of a client by inspecting the messages
     it exchanges with the server.

    Args:
        messages_metadata (dict): A dictionary containing the metadata of the messages exchanged between
            the attacked client and the server during a federated training process.
            The dictionary should have the following structure:
                {
                    "server": {
                        <round_id_1>: "<path/to/server/checkpoints_1>",
                        <round_id_2>: "<path/to/server/checkpoints_2>",
                        ...
                    },
                    "client": {
                        <round_id_1>: "<path/to/client/checkpoints_1>",
                        <round_id_2>: "<path/to/client/checkpoints_2>",
                        ...
                    }
                }
        model_init_fn: Function to initialize the federated learning model.
        gradient_prediction_trainer (Trainer): Trainer for the gradient prediction model.
        optimizer_name (str): Name of the optimizer to use (default is "sgd").
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): momentum used with the optimizer.
        weight_decay (float): weight decay used with the optimizer.
        dataset (torch.utils.data.Dataset): The dataset of the attacked client.
        logger: The logger for recording simulation logs.
        log_freq (int): Frequency for logging simulation information.
        rng: A random number generator, which can be used for any randomized behavior within the attack.
            If None, a default random number generator will be created.

    Attributes:
        messages_metadata (dict): A dictionary containing the metadata of the messages exchanged between
            the attacked client and the server during a federated training process.
            The dictionary should have the following structure:
                {
                    "global": {
                        <round_id_1>: "<path/to/global/checkpoints_1>",
                        <round_id_2>: "<path/to/global/checkpoints_2>",
                        ...
                    },
                    "local": {
                        <round_id_1>: "<path/to/local/checkpoints_1>",
                        <round_id_2>: "<path/to/local/checkpoints_2>",
                        ...
                    }
                }
        model_init_fn: Function to initialize the federated learning model.
        gradient_prediction_trainer (Trainer): Trainer for the gradient prediction model.
        round_ids (list): List of round IDs extracted from the provided messages metadata.
        last_round (int): Integer representing the last round.
        last_round_id (str): String representing the ID of the last round
        dataset (torch.utils.data.Dataset): The dataset of the attacked client.
        gradients_dataset (torch.utils.data.TensorDataset): contains tensors of global model parameters
                and pseudo-gradients for each round.
        gradients_loader (torch.utils.data.DataLoader): iterates over tensors of global model parameters
                and pseudo-gradients for each round.
        device (str or torch.Device): Device on which to perform computations.
        n_samples (int): Number of samples
        optimizer_name (str): Name of the optimizer to use (default is "sgd").
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): momentum used with the optimizer.
        weight_decay (float): weight decay used with the optimizer.
        optimizer (torch.optim.Optimizer):
        global_models_dict (dict): dictionary mapping round ids to global models
        pseudo_gradients_dict (dict): dictionary mapping round ids to pseudo-gradients
        reconstructed_model_params (torch.Tensor): Reconstructed model parameters
        logger: The logger for recording simulation logs.
        log_freq (int): Frequency for logging simulation information.
        rng: A random number generator, which can be used for any randomized behavior within the attack.
            If None, a default random number generator will be created.
        dataset: The dataset of the attacked client.

    Methods:
        _get_round_ids():
            Retrieve the round IDs from the provided messages metadata.

        _get_model_at_round(round_id, mode="global"):
            Retrieve the model at a specific communication round.

        _get_models_dict(mode="global"):
            Retrieve a dictionary of models at different communication rounds.

        _compute_pseudo_gradient_at_round(round_id):
            Compute the pseudo-gradient associated with one communication round between the client and the server.

        _compute_pseudo_gradients_dict():
            Compute pseudo-gradients for all communication rounds and store them in a dictionary.

        _initialize_gradients_dataset():
            Construct and initialize a dataset iterating (across rounds) over the model
            parameters and the pseudo-gradients.

        _initialize_reconstructed_model_params():
            Initialize and return the parameters of a reconstructed model.

        _initialize_optimizer():
            Initialize and return an optimizer for training the reconstructed model.

        _freeze_gradient_predictor():
            Freezes the gradient predictor model by setting the `requires_grad` attribute to `False`
            for all its parameters.

        fit_gradient_predictor(num_rounds):
            Trains the gradient predictor for a specified number of rounds using the provided gradients loader.

        execute_attack(num_iterations):
            Execute the federated learning attack on the provided dataset.

        evaluate_attack():
            Evaluate the success of the federated learning attack on the provided dataset.
    """
    def __init__(
            self, messages_metadata, model_init_fn, gradient_prediction_trainer,
            optimizer_name, learning_rate, momentum, weight_decay, dataset,
            logger, log_freq, rng=None
    ):
        """
        Initialize the AttributeInferenceAttack.

        Parameters:
        - messages_metadata: Metadata containing information about communication rounds.
        - model_init_fn: Function to initialize the federated learning model.
        - gradient_prediction_trainer (Trainer): Trainer for the gradient prediction model.
        - dataset: Federated learning dataset.
        - optimizer_name (str): Name of the optimizer to use (default is "sgd").
        - learning_rate (float): Learning rate for the optimizer.
        - momentum (float): momentum used with the optimizer.
        - weight_decay (float): weight decay used with the optimizer.
        - logger: Logger for recording metrics.
        - log_freq (int): Frequency for logging simulation information.
        - rng: Random number generator for reproducibility.
        """

        self.messages_metadata = messages_metadata

        self.model_init_fn = model_init_fn

        self.gradient_prediction_trainer = gradient_prediction_trainer

        self.device = gradient_prediction_trainer.device

        self.dataset = dataset

        self.logger = logger
        self.log_freq = log_freq

        self.rng = rng if rng is not None else np.random.default_rng()

        self.optimizer_name = optimizer_name

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.round_ids = self._get_round_ids()
        self.last_round = max(map(int, self.round_ids))
        self.last_round_id = str(self.last_round)

        self.n_samples = len(self.dataset)

        self.global_models_dict = self._get_models_dict(mode="global")

        self.pseudo_gradients_dict = self._compute_pseudo_gradients_dict()

        self.gradients_dataset = self._initialize_gradients_dataset()

        self.gradients_loader = DataLoader(self.gradients_dataset, batch_size=len(self.round_ids), shuffle=True)

        self.reconstructed_model_params = self._initialize_reconstructed_model_params()

        self.reconstructed_model = self.model_init_fn().to(self.device)

        self.optimizer = self._initialize_optimizer()

    def _get_round_ids(self):
        """
        Retrieve the round IDs from the provided messages metadata.

        Returns:
            list: List of round IDs.
        """

        assert set(self.messages_metadata["global"].keys()) == set(self.messages_metadata["local"].keys()), \
            "Global and local round ids do not match!"

        return list(self.messages_metadata["global"].keys())

    def _get_model_at_round(self, round_id, mode="global"):
        """
        Retrieve the model at a specific communication round.

        Parameters:
        - round_id (int): The identifier of the communication round.
        - mode (str): The mode specifying whether to retrieve a local or global model (default is "global").

        Returns:
        - torch.nn.Module: The model at the specified communication round.
        """

        assert mode in {"local", "global"}, f"`mode` should be 'local' or 'global', not {mode}"

        model_chkpts = torch.load(self.messages_metadata[mode][round_id])["model_state_dict"]
        model = self.model_init_fn()
        model.load_state_dict(model_chkpts)

        return model

    def _get_models_dict(self, mode="global"):
        """
        Retrieve a dictionary of models at different communication rounds.

        Parameters:
        - mode (str): The mode specifying whether to retrieve local or global models (default is "global").

        Returns:
        - dict: A dictionary where keys are communication round identifiers, and values are corresponding models.
        """
        assert mode in {"local", "global"}, f"`mode` should be 'local' or 'global', not {mode}"

        models_dict = dict()
        for round_id in self.round_ids:
            models_dict[round_id] = get_param_tensor(self._get_model_at_round(round_id=round_id, mode=mode))

        return models_dict

    def _compute_pseudo_gradient_at_round(self, round_id):
        """
        Compute the pseudo-gradient associated with one communication round between the client and the server.

        The pseudo-gradient is defined as the difference between the global model and the model
        resulting from the client's local update.

        Parameters
        ----------
        round_id : int
            The identifier of the communication round for which to compute the pseudo-gradient.

        Returns
        -------
        torch.Tensor
            A flattened tensor representing the pseudo-gradient, computed as the difference between
            the global model parameters and the local model parameters after the client's update.
        """

        global_model = self._get_model_at_round(round_id=round_id, mode="global")
        local_model = self._get_model_at_round(round_id=round_id, mode="local")

        global_param_tensor = get_param_tensor(global_model)
        local_param_tensor = get_param_tensor(local_model)

        return global_param_tensor - local_param_tensor

    def _compute_pseudo_gradients_dict(self):
        """
        Compute pseudo-gradients for all communication rounds and store them in a dictionary.

        Returns:
        - dict: A dictionary where keys are communication round identifiers,
            and values are corresponding pseudo-gradients.
        """
        pseudo_gradients_dict = dict()
        for round_id in self.round_ids:
            pseudo_gradients_dict[round_id] = self._compute_pseudo_gradient_at_round(round_id=round_id)

        return pseudo_gradients_dict

    def _initialize_gradients_dataset(self):
        """
        Construct and initialize a PyTorch dataset by iterating across rounds over the model parameters
        and the corresponding pseudo-gradients.

        Returns:
            TensorDataset: A torch.utils.data.TensorDataset containing tensors of global model parameters
                and pseudo-gradients for each round.
        """
        global_models_list = []
        pseudo_gradients_list = []

        for round_id in self.round_ids:
            global_models_list.append(self.global_models_dict[round_id])
            pseudo_gradients_list.append(self.pseudo_gradients_dict[round_id])

        global_models_tensor = torch.stack(global_models_list)
        pseudo_gradients_tensor = torch.stack(pseudo_gradients_list)

        return TensorDataset(global_models_tensor, pseudo_gradients_tensor)

    def _initialize_reconstructed_model_params(self):
        """
        Initialize and return the parameters of a reconstructed model.

        Returns:
            torch.Tensor: Tensor containing the initialized model parameters.
        """
        return get_param_tensor(self.model_init_fn()).clone().detach().requires_grad_(True).to(self.device)

    def _initialize_optimizer(self):
        """
        Initialize and return an optimizer for training the reconstructed model.

        Returns:
            torch.optim.Optimizer: The initialized optimizer for the reconstructed model.
        """
        if self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                [self.reconstructed_model_params],
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Optimizer '{self.optimizer_name}' is not implemented"
            )

        return optimizer

    def _freeze_gradient_predictor(self):
        """
        Freezes the gradient predictor model by setting the `requires_grad` attribute to `False` for all its parameters.

        Returns:
            None
        """
        self.gradient_prediction_trainer.freeze_model()

    def fit_gradient_predictor(self, num_iterations):
        """
        Train the gradient predictor for a specified number of rounds using the provided gradients loader.

        Args:
            num_iterations (int): The number of training rounds for the gradient predictor.

        Returns:
            None
        """
        for c_iteration in tqdm(range(num_iterations), leave=False):
            self.gradient_prediction_trainer.fit_epoch(self.gradients_loader)
            loss_val, metric_val = self.gradient_prediction_trainer.evaluate_loader(
                self.gradients_loader)

            self.logger.add_scalar("Gradient Estimation Loss", loss_val, c_iteration)

            if c_iteration % self.log_freq == 0:
                logging.info("+" * 50)
                logging.info(f"Iteration {c_iteration}: Gradient Estimation Loss: {loss_val:4f}")
                logging.info("+" * 50)

    def reconstruct_local_model(self, num_iterations):
        """
        Reconstructs a local model by iteratively updating its parameters based on estimated gradients.

        Args:
            num_iterations (int): Number of optimization iterations for reconstructing the local model.

        Returns:
            torch.nn.Module:
                The reconstructed local model.
        """

        for c_iteration in tqdm(range(num_iterations), leave=False):
            self.optimizer.zero_grad()

            loss = torch.linalg.vector_norm(
                self.gradient_prediction_trainer.model(self.reconstructed_model_params),
                ord=2
            )

            loss.backward()

            self.optimizer.step()

            loss_val = loss.item()
            self.logger.add_scalar("Estimated Gradient Norm", loss_val, c_iteration)

            if c_iteration % self.log_freq == 0:
                logging.info("+" * 50)
                logging.info(f"Iteration {c_iteration}: Estimated Gradient Norm: {loss_val:4f}")
                logging.info("+" * 50)

        reconstructed_model = self.model_init_fn().to(self.device)

        set_param_tensor(
            model=reconstructed_model,
            param_tensor=self.reconstructed_model_params.detach(),
            device=self.device
        )

        reconstructed_model.eval()

        return reconstructed_model

    def execute_attack(self, num_iterations):
        """
        Executes local model reconstruction attack by performing the following steps:

        1. Fits a gradient predictor using a specified number of iterations.
        2. Freezes the gradient predictor to prevent further training.
        3. Reconstructs a local model using the fitted gradient predictor. Updates the 'reconstructed_model' attribute.

        Parameters:
        - num_iterations (int): The number of iterations used for fitting the gradient predictor
                              and reconstructing the local model.

        Returns:
        torch.nn.Module:
            reconstructed_model: The reconstructed local model after the attack.
        """
        self.fit_gradient_predictor(num_iterations=num_iterations)
        self._freeze_gradient_predictor()
        self.reconstructed_model = self.reconstruct_local_model(num_iterations=num_iterations)

        return self.reconstructed_model

    def evaluate_attack(self, reference_model, dataloader, task_type, epsilon=1e-10):
        """
        Calculate Jensen-Shannon Divergence (JSD) between the output distributions of
        the constructed model and a reference model.

        Parameters:
        - reference_model (torch.nn.Module): The reference model for comparison.
        - dataloader (torch.utils.data.DataLoader): DataLoader providing input data for both models.
        - task_type (str): Type of the task, one of "binary_classification", "classification", or "regression".
        - epsilon (float): A small value added to the probabilities to avoid division by zero, default is 1e-10.

        Returns:
        - jsd_value (float): Jensen-Shannon Divergence between the output distributions of the two models.
        """
        model = self.reconstructed_model

        reference_outputs_list = []
        outputs_list = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                reference_outputs_list.append(reference_model(inputs))
                outputs_list.append(model(inputs))

        reference_outputs = torch.cat(reference_outputs_list)
        outputs = torch.cat(outputs_list)

        if task_type == "binary_classification":
            reference_outputs = torch.sigmoid(reference_outputs)
            outputs = torch.sigmoid(outputs)
            distribution_type = 'bernoulli'

        elif task_type == "classification":
            reference_outputs = torch.softmax(reference_outputs)
            outputs = torch.softmax(outputs)
            distribution_type = 'multinomial'

        elif task_type == "regression":
            distribution_type = 'gaussian'

        else:
            raise NotImplementedError(
                "Invalid distribution type."
                "Possible values: 'binary_classification', 'classification', 'regression'."
            )

        score = jsd(reference_outputs, outputs, distribution_type=distribution_type, epsilon=epsilon)
        return float(score)
