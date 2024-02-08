import os
import logging

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from ..utils import *

from .utils import *


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
                    "global": {
                        <round_id_1>: "<path/to/server/checkpoints_1>",
                        <round_id_2>: "<path/to/server/checkpoints_2>",
                        ...
                    },
                    "local": {
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
        gradient_oracle (GradientOracle): computing and retrieving gradients of a given model's parameters.
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
            logger, log_freq, rng=None,
            gradient_oracle=None
    ):
        """
        Initialize the LocalModelReconstructionAttack.

        Parameters:
        - messages_metadata: Metadata containing information about communication rounds.
        - model_init_fn: Function to initialize the federated learning model.
        - gradient_prediction_trainer (Trainer): Trainer for the gradient prediction model.
        - gradient_oracle (GradientOracle): computing and retrieving gradients of a given model's parameters.
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

        self.gradient_oracle = gradient_oracle

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

        self.reconstructed_model = self._init_reconstructed_model()

        self.reconstructed_model_params = get_param_tensor(self.reconstructed_model).clone().detach().to(self.device)

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

    def _init_reconstructed_model(self):
        """
        Initialize a reconstructed model by calling the provided model initialization function.

        Returns:
            torch.nn.Module: The initialized model with gradients set to zero.
        """
        model = self._get_model_at_round(self.last_round_id, mode="local").to(self.device)

        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.data.zero_()

        return model

    def _initialize_optimizer(self):
        """
        Initialize and return an optimizer for training the reconstructed model.

        Returns:
            torch.optim.Optimizer: The initialized optimizer for the reconstructed model.
        """
        if self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                [param for param in self.reconstructed_model.parameters() if param.requires_grad],
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
            A tuple (loss, metric), where loss is the average value of the loss
            function over all batches, and average_metric is the average computed evaluation metric.
        """
        for c_iteration in tqdm(range(num_iterations), leave=False):
            self.gradient_prediction_trainer.fit_epoch(self.gradients_loader)
            loss, metric = self.gradient_prediction_trainer.evaluate_loader(self.gradients_loader)

            self.logger.add_scalar("Gradient Estimation Loss", loss, c_iteration)

            if c_iteration % self.log_freq == 0:
                logging.debug("+" * 50)
                logging.debug(f"Iteration {c_iteration}: Gradient Estimation Loss: {loss:4f}")
                logging.debug("+" * 50)

        loss, metric = self.gradient_prediction_trainer.evaluate_loader(self.gradients_loader)

        return loss, metric

    def verify_gradient_predictor(self, scaling_coeff):
        # TODO (OTHMANE): clarify docs for this function
        for round_id in self.round_ids:
            model_params = self.global_models_dict[round_id].to(self.device)

            pseudo_gradient = scaling_coeff * self.pseudo_gradients_dict[round_id].to(self.device)

            oracle_gradient = self.gradient_oracle.predict_gradient(model_params).detach()

            predicted_gradient = scaling_coeff * self.gradient_prediction_trainer.model(model_params).detach()

            err = F.mse_loss(predicted_gradient, oracle_gradient)

            logging.debug(
                f"params: {model_params} | oracle: {oracle_gradient} | pseudo: {pseudo_gradient} |"
                f" predicted: {predicted_gradient} | err: {err}"
            )

    def reconstruct_local_model(
            self, num_iterations, use_gradient_oracle, save_dir=None, save_freq=1, debug=False, scaling_coeff=None
    ):
        """
        Reconstructs a local model by iteratively updating its parameters based on estimated gradients.

        Args:
            num_iterations (int): Number of optimization iterations for reconstructing the local model.
            use_gradient_oracle (bool): If `True`, use gradient oracle; otherwise, use gradient predictor.
            save_dir (str or None): Directory to save reconstructed models. If `None`, models will not be saved.
            save_freq (int): Frequency for saving reconstructed models.
            debug (bool): If `True`, enables debug mode for additional logging and checks. Default is `False`.
            scaling_coeff (float or None): A scaling coefficient applied to the predicted gradient
                if debug mode is enabled.

        Returns:
            Dict[str: str]: A dictionary containing metadata for saved reconstructed models,
             where keys are iteration numbers and values are corresponding file paths.
             Returns an empty dictionary if no models are saved.

        """
        reconstructed_models_metadata_dict = dict()

        for c_iteration in tqdm(range(num_iterations), leave=False):
            self.optimizer.zero_grad(set_to_none=False)

            if use_gradient_oracle and self.gradient_oracle is not None:
                gradient = self.gradient_oracle.predict_gradient(self.reconstructed_model_params)
            elif use_gradient_oracle:
                logging.warning("Gradient Oracle is not provided. Gradient predictor is used!")
                gradient = self.gradient_prediction_trainer.model(self.reconstructed_model_params)
            else:
                gradient = self.gradient_prediction_trainer.model(self.reconstructed_model_params)

            gradient = gradient.detach()

            if debug and scaling_coeff is not None:
                oracle_gradient = self.gradient_oracle.predict_gradient(self.reconstructed_model_params)
                oracle_gradient = oracle_gradient.detach()

                predicted_gradient = self.gradient_prediction_trainer.model(self.reconstructed_model_params)
                predicted_gradient = predicted_gradient.detach()

                predicted_gradient = scaling_coeff * predicted_gradient

                err = F.mse_loss(predicted_gradient, oracle_gradient)

                logging.debug(
                    f"params: {self.reconstructed_model_params} | oracle: {oracle_gradient} | "
                    f" predicted: {predicted_gradient} | err: {err}"
                )

            set_grad_tensor(model=self.reconstructed_model, grad_tensor=gradient, device=self.device)

            self.optimizer.step()

            self.reconstructed_model_params = (
                get_param_tensor(self.reconstructed_model).clone().detach().to(self.device)
            )

            loss = torch.linalg.vector_norm(gradient, ord=2).item()

            self.logger.add_scalar("Estimated Gradient Norm", loss, c_iteration)

            logging.debug(f"reconstructed params: {self.reconstructed_model_params}")

            log_flag = c_iteration % self.log_freq == 0
            save_flag = c_iteration % save_freq == 0

            if log_flag:
                logging.debug("+" * 50)
                logging.debug(f"Iteration {c_iteration}: Estimated Gradient Norm: {loss:4f}")
                logging.debug("+" * 50)

            if save_flag:
                logging.debug("+" * 50)
                logging.info(f"Save reconstructed model for iteration {c_iteration}..")
                checkpoint = {'model_state_dict': self.reconstructed_model.state_dict()}

                save_path = os.path.join(save_dir, f"{c_iteration}.pt")
                save_path = os.path.abspath(save_path)
                torch.save(checkpoint, save_path)

                reconstructed_models_metadata_dict[f"{c_iteration}"] = save_path

                logging.debug("+" * 50)

        return reconstructed_models_metadata_dict

    def execute_attack(
            self, num_iterations, use_gradient_oracle=False, save_dir=None, save_freq=1, debug=False, scaling_coeff=None
    ):
        """
        Executes local model reconstruction attack by performing the following steps:

        1. Fits a gradient predictor using a specified number of iterations.
        2. Freezes the gradient predictor to prevent further training.
        3. Reconstructs a local model using the fitted gradient predictor. Updates the 'reconstructed_model' attribute.

        Parameters:
        - num_iterations (int): The number of iterations used for fitting the gradient predictor
                              and reconstructing the local model.
        - use_gradient_oracle (bool): If `True`, use gradient oracle; otherwise, use gradient predictor.
        - save_dir (str): Directory to save reconstructed models. If `None`, models will not be saved.
        - save_freq (int): Frequency for saving reconstructed models.
        - debug (bool): If `True`, enables debug mode for additional logging and checks. Default is `False`.
        - scaling_coeff (float or None): A scaling coefficient applied to the predicted gradient
                if debug mode is enabled.

        Returns:
            * Dict[str: str]: A dictionary containing metadata for saved reconstructed models,
                where keys are iteration ids and values are corresponding file paths.
        """

        estimation_loss, estimation_metric = self.fit_gradient_predictor(num_iterations=num_iterations)

        logging.info(f"Gradient Estimation Loss: {estimation_loss:4f} | Metric: {estimation_metric:4f}")

        self._freeze_gradient_predictor()

        if debug and scaling_coeff is not None:
            self.verify_gradient_predictor(scaling_coeff=scaling_coeff)

        # Note: It is that reconstructed_models_metadata_dict is an empty dictionary.
        reconstructed_models_metadata_dict = self.reconstruct_local_model(
            num_iterations=num_iterations, use_gradient_oracle=use_gradient_oracle,
            save_dir=save_dir, save_freq=save_freq, debug=debug, scaling_coeff=scaling_coeff
        )

        if not reconstructed_models_metadata_dict:
            # Add the final reconstructed model to the metadata if the metadata is empty
            checkpoint = {'model_state_dict': self.reconstructed_model.state_dict()}
            save_path = os.path.join(save_dir, f"{num_iterations}.pt")
            save_path = os.path.abspath(save_path)
            torch.save(checkpoint, save_path)

            reconstructed_models_metadata_dict[f"{num_iterations-1}"] = save_path

        return reconstructed_models_metadata_dict

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
        return model_jsd(
            self.reconstructed_model, reference_model, dataloader=dataloader, task_type=task_type,
            device=self.device, epsilon=epsilon
        )


class GradientOracle:
    """
    A class for computing and retrieving gradients of a given model's parameters with respect to a specified criterion.

    Attributes:
    - model_init_fn (function): A function that initializes the model architecture.
    - dataset (torch.utils.data.Dataset): The dataset used for training the model.
    - is_binary_classification (bool): A flag indicating whether the task is binary classification.
    - criterion (torch.nn.Module): The loss criterion used for training the model.
    - device (torch.device): The device (CPU or GPU) on which the model and computations are performed.
    - model (torch.nn.Module): The initialized model.
    - true_features (torch.Tensor): Tensor containing all features from the dataset.
    """

    def __init__(self, model_init_fn, dataset, is_binary_classification, criterion, device):
        """
        Initialize the GradientOracle.

        Parameters:
        - model_init_fn (callable): A function that initializes the model.
        - dataset: The dataset containing features and labels.
        - is_binary_classification (bool): Indicates whether the task is binary classification.
        - criterion: The loss criterion used for computing gradients.
        - device: The device (e.g., 'cpu' or 'cuda') on which the computations will be performed.
        """

        self.model_init_fn = model_init_fn
        self.dataset = dataset

        self.criterion = criterion
        self.is_binary_classification = is_binary_classification

        self.device = device

        self.model = self.model_init_fn()

        self.true_features, self.true_labels = self._get_all_features()

        self.true_features = self.true_features.to(self.device).type(torch.float32)
        self.true_labels = self.true_labels.to(self.device)

        if self.is_binary_classification:
            self.true_labels = self.true_labels.type(torch.float32)

    def _get_all_features(self):
        """
        Retrieve all features and labels from the dataset.

        Returns:
        - Tuple of torch.Tensors: A tuple containing tensors representing all features and labels in the dataset.
        """
        return get_all_features(self.dataset)

    def _set_model_parameters(self, params):
        """
        Set the model parameters to the given tensor.

        Parameters:
        - params: The tensor containing model parameters.
        """
        set_param_tensor(
            model=self.model,
            param_tensor=params,
            device=self.device
        )

    def predict_gradient(self, params):
        """
         Predict the gradients of the model parameters based on the given parameters.

         Parameters:
         - params: The tensor containing model parameters.

         Returns:
         - torch.Tensor: The computed gradients.
         """
        self._set_model_parameters(params)

        self.model.zero_grad()

        predictions = self.model(self.true_features)

        if predictions.shape[0] != 1:
            predictions = predictions.squeeze()
        else:
            predictions = predictions.squeeze(dim=tuple(predictions.shape[1:]))

        loss = self.criterion(predictions, self.true_labels)

        loss.backward()

        return get_grad_tensor(self.model)
