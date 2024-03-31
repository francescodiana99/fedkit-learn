import os
import shutil
from abc import ABC, abstractmethod

import logging

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from scipy.special import logit

from tqdm import tqdm

from ..utils import *

from .utils import *

import pandas as pd


EPSILON = 1e-5


class BaseAttributeInferenceAttack(ABC):
    """
    Class representing an attribute inference attack in federated learning.
    This attack aims to infer sensitive attributes of clients from federated learning updates or models.

    Args:
        dataset (torch.utils.data.Dataset): The dataset of the attacked client.
        sensitive_attribute_id (int): Index of the sensitive attribute in the dataset.
        sensitive_attribute_type (str): Type of the sensitive attribute ("binary", "categorical", or "numerical").
        initialization (str): Strategy used to initialize the sensitive attribute. Possible values are "normal".
        device (str or torch.Device): Device on which to perform computations.
        criterion (torch.nn.Criterion): Loss criterion.
        is_binary_classification (bool): True if the federated learning task is binary classification.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_name (str): Name of the optimizer to use (default is "sgd").
        success_metric: Metric to evaluate the success of the attack.
        rng: A random number generator, which can be used for any randomized behavior within the attack.
            If None, a default random number generator will be created.

    Attributes:
        dataset (torch.utils.data.Dataset): The dataset of the attacked client.
        sensitive_attribute_id (int): Index of the sensitive attribute in the dataset.
        sensitive_attribute_type (str): Type of the sensitive attribute ("binary", "categorical", or "numerical").
        initialization (str): Strategy used to initialize the sensitive attribute. Possible values are "normal".
        device (str or torch.Device): Device on which to perform computations.
        criterion: Loss criterion for the attack.
        is_binary_classification (bool): True if the federated learning task is binary classification.
        learning_rate (float): Learning rate for the optimizer.
        optimizer_name (str): Name of the optimizer to use (default is "sgd").
        success_metric: Metric to evaluate the success of the attack.
        rng: A random number generator, which can be used for any randomized behavior within the attack.
            If None, a default random number generator will be created.

        n_samples (int): Number of samples
        true_features (torch.Tensor): true features tensor; shape=(n_samples, n_features).
        predicted_features (torch.Tensor): predicted features tensor; shape=(n_samples, n_features).
        true_labels (torch.Tensor): true labels' tensor.

    Methods:
        execute_attack(num_iterations):
            Execute the federated learning attack on the provided dataset.

        evaluate_attack():
            Evaluate the success of the federated learning attack on the provided dataset.

    """
    def __init__(self, dataset, sensitive_attribute_id, sensitive_attribute_type, initialization, device,
                 criterion, is_binary_classification, learning_rate, optimizer_name, success_metric,
                 rng=None, torch_rng=None):

        self.dataset = dataset
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self.torch_rng = torch_rng if torch is not None else torch.Generator()

        self.sensitive_attribute_id = sensitive_attribute_id

        self.sensitive_attribute_type = sensitive_attribute_type

        if self.sensitive_attribute_type not in {"binary", "categorical", "numerical"}:
            raise ValueError(
                f'{self.sensitive_attribute_type} is not a supported type for the sensitive attribute.'
                f'Possible are: "binary", "categorical", "numerical"'
            )

        if self.sensitive_attribute_type == "categorical":
            raise NotImplementedError(
                "Attribute Inference Attack is not yet implemented for categorical variables!"
            )

        self.success_metric = success_metric

        self.initialization = initialization

        self.criterion = criterion
        self.is_binary_classification = is_binary_classification

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.n_samples = len(self.dataset)
        self.true_features, self.true_labels = self._get_all_features()

        self.true_features = self.true_features.to(self.device).type(torch.float32)
        self.true_labels = self.true_labels.to(self.device)

        if self.is_binary_classification:
            self.true_labels = self.true_labels.type(torch.float32).unsqueeze(1)

        self.num_classes = self._compute_num_sensitive_classes()

        self.predicted_features = self.true_features.clone()

        self.sensitive_attribute_interval = self._get_sensitive_attribute_interval()

        self.predicted_features[:, self.sensitive_attribute_id] = 0.

    def _get_all_features(self):
        """
        Retrieve all features and labels from the federated learning dataset.

        Returns:
        - Tuple of torch.Tensors: A tuple containing tensors representing all features and labels in the dataset.
        """
        return get_all_features(self.dataset)

    def _compute_num_sensitive_classes(self):
        if self.sensitive_attribute_type == "binary":
            num_classes = 1  # num_classes is only used to initialize logits
        elif self.sensitive_attribute_type == "categorical":
            num_classes = torch.unique(self.true_features[:, self.sensitive_attribute_id]).numel()
        elif self.sensitive_attribute_type == "numerical":
            num_classes = 1
        else:
            raise NotImplementedError(
                f'{self.sensitive_attribute_type} is not a supported type for the sensitive attribute.'
                f'Possible are: "binary", "categorical", "numerical"'
            )

        return num_classes

    def _get_sensitive_attribute_interval(self):
        """
        Calculate the interval of the sensitive attribute values in the dataset.

        Returns:
        - Tuple of float: A tuple containing the lower and upper bounds of the sensitive attribute values.
        """
        lower_bound = self.true_features[:, self.sensitive_attribute_id].min()
        upper_bound = self.true_features[:, self.sensitive_attribute_id].max()
        return lower_bound, upper_bound

    def _init_optimizer(self, sensitive_attribute):
        if self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                [sensitive_attribute],
                lr=self.learning_rate
            )
        else:
            raise NotImplementedError(
                f"Optimizer '{self.optimizer_name}' is not implemented."
            )

        return optimizer

    @abstractmethod
    def execute_attack(self, num_iterations):
        """
        Execute the federated learning attack on the provided dataset.

        Parameters:
        - num_iterations (int): The number of iterations to perform the attack.
        """
        pass

    @abstractmethod
    def evaluate_attack(self):
        """
        Evaluate the success of the federated learning attack on the provided dataset.

        Returns:
        - torch.Tensor: The success metric value.
        """
        pass


class AttributeInferenceAttack(BaseAttributeInferenceAttack):
    """
    Class representing an attribute inference attack in federated learning.
    This attack aims to infer sensitive attributes of clients from federated learning updates.

    Implementation based on the technique presented in (https://arxiv.org/abs/2108.06910)_

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
        model_init_fn (Callable): Function to initialize the federated learning model.
        gumbel_temperature (float): non-negative scalar temperature used for Gumbel-Softmax distribution.
        gumbel_threshold (float): non-negative scalar, between 0 and 1, used as a threshold in the binary case.
        logger: The logger for recording simulation logs.
        log_freq (int): Frequency for logging simulation information.

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
        round_ids (list): List of round IDs extracted from the provided messages metadata.
        model_init_fn: Function to initialize the federated learning model.
        optimizer (torch.optim.Optimizer):
        n_samples (int): Number of samples
        sensitive_attribute_interval (tuple): lower and upper bounds for the sensitive attribute interval.
        sensitive_attribute_logits (torch.Tensor): logits of the sensitive attribute.
        gumbel_temperature (float): non-negative scalar temperature used for Gumbel-Softmax distribution.
        gumbel_threshold (float): non-negative scalar, between 0 and 1, used as a threshold in the binary case.
        global_models_dict (dict): dictionary mapping round ids to global models.
        pseudo_gradients_dict (dict): dictionary mapping round ids to pseudo-gradients.
        logger: The logger for recording simulation logs.
        log_freq (int): Frequency for logging simulation information.

    Methods:
        _get_model_at_round(round_id, mode="global"):
            Retrieve the model at a specific communication round.

        _get_models_dict(mode="global"):
            Retrieve a dictionary of models at different communication rounds.

        _compute_pseudo_gradient_at_round(round_id):
            Compute the pseudo-gradient associated with one communication round between the client and the server.

        _compute_pseudo_gradients_dict():
            Compute pseudo-gradients for all communication rounds and store them in a dictionary.

    """
    def __init__(
            self, messages_metadata, dataset, sensitive_attribute_id, sensitive_attribute_type, initialization,
            device, model_init_fn, criterion, is_binary_classification, learning_rate, optimizer_name, success_metric,
            logger, log_freq, gumbel_temperature=1.0, gumbel_threshold=0.5, rng=None, torch_rng=None
    ):
        """
        Initialize the AttributeInferenceAttack.

        Parameters:
        - messages_metadata: Metadata containing information about communication rounds.
        - dataset: Federated learning dataset.
        - sensitive_attribute_id: Index of the sensitive attribute in the dataset.
        - sensitive_attribute_type: Type of the sensitive attribute ("numerical" or "binary").
        - initialization: Strategy used to initialize the sensitive attribute. Possible are "uniform" and "fixed".
        - device (str or torch.Device): Device on which to perform computations.
        - model_init_fn: Function to initialize the federated learning model.
        - criterion: Loss criterion for the attack.
        - is_binary_classification: True if the federated learning task is binary classification.
        - learning_rate: Learning rate for the optimizer.
        - optimizer_name: Name of the optimizer to use (default is "sgd").
        - success_metric: Metric to evaluate the success of the attack.
        - logger: Logger for recording metrics.
        - log_freq (int): Frequency for logging simulation information.
        - gumbel_temperature (float): non-negative scalar temperature used for Gumbel-Softmax distribution
        - gumbel_threshold (float): non-negative scalar, between 0 and 1, used as a threshold in the binary case
        - rng: Random number generator for reproducibility.
        - torch_rng (torch.Generator): Random number generator for reproducibility.
        """
        super(AttributeInferenceAttack, self).__init__(
            dataset=dataset,
            sensitive_attribute_id=sensitive_attribute_id,
            sensitive_attribute_type=sensitive_attribute_type,
            initialization=initialization,
            device=device,
            criterion=criterion,
            is_binary_classification=is_binary_classification,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            success_metric=success_metric,
            rng=rng,
            torch_rng=torch_rng
        )
        self.messages_metadata = messages_metadata

        self.gumbel_temperature = gumbel_temperature
        self.gumbel_threshold = gumbel_threshold

        self.logger = logger
        self.log_freq = log_freq

        self.model_init_fn = model_init_fn

        self.round_ids = self._get_round_ids()

        self.sensitive_attribute_logits = self._init_sensitive_attribute_logits()

        self.sensitive_attribute = self._sample_sensitive_attribute(deterministic=True)

        self.optimizer = self._init_optimizer(self.sensitive_attribute_logits)

        self.global_models_dict = self._get_models_dict(mode="global")

        self.pseudo_gradients_dict = self._compute_pseudo_gradients_dict()

    def _get_round_ids(self):
        """
        Retrieve the round IDs from the provided messages metadata.

        Returns:
            list: List of round IDs.
        """

        assert set(self.messages_metadata["global"].keys()) == set(self.messages_metadata["local"].keys()), \
            "Global and local round ids do not match!"

        return list(self.messages_metadata["global"].keys())

    def _init_sensitive_attribute_logits(self):
        """
        Initialize the logits of the sensitive attribute.

        Returns:
        - torch.Tensor: A tensor representing the initialized logits of the sensitive attribute with gradients enabled.
        """
        if self.initialization == "normal":
            logits = torch.randn(size=(self.n_samples, self.num_classes), generator=self.torch_rng)
        else:
            raise NotImplementedError(
                f"{self.initialization} is not a valid initialization strategy. "
                f"Possible are 'normal'."
            )

        logits = logits.clone().detach().requires_grad_(True).to(self.device)

        return logits

    def _sample_sensitive_attribute(self, hard=True, deterministic=False):
        """
        Get the value of sensitive attributes based on the current value of the logits.

        This function samples the sensitive attributes according to their distribution, which is determined by the
        type of sensitive attribute. The supported types include "binary", "categorical", and "numerical".

        For "binary" or "categorical" sensitive attributes, the function uses the Gumbel Softmax relaxation to
        obtain a differentiable approximation of the discrete sampling process. For "numerical" sensitive attributes,
        the logits are directly used as the sampled value.

         Parameters:
        - hard (bool, optional): if True, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
        - deterministic (bool, optional): If True, sample the variable with the maximum probability deterministically.
          Defaults to True.

        Returns:
        - torch.Tensor: A tensor representing the sampled sensitive attributes. The shape is `(self.num_samples,)` for
          multiple samples, or a scalar tensor for a single sample.

        Raises:
        - ValueError: If the specified sensitive attribute type is not one of the supported types
          ("binary", "categorical", "numerical").
        """
        if self.sensitive_attribute_type == "binary" or self.sensitive_attribute_type == "categorical":
            if deterministic:
                sensitive_attribute = get_most_probable_class(
                    logits=self.sensitive_attribute_logits,
                    threshold=logit(self.gumbel_threshold),
                    dim=-1
                )

            else:
                sensitive_attribute = gumbel_softmax(
                    binary=(self.sensitive_attribute_type == "binary"),
                    logits=self.sensitive_attribute_logits,
                    tau=self.gumbel_temperature,
                    threshold=self.gumbel_threshold,
                    generator=self.torch_rng,
                    hard=hard,
                    dim=-1
                )

        elif self.sensitive_attribute_type == "numerical":
            sensitive_attribute = self.sensitive_attribute_logits

        else:
            raise ValueError(
                f'{self.sensitive_attribute_type} is not a supported type for the sensitive attribute.'
                f'Possible are: "binary", "categorical", "numerical"'
            )

        sensitive_attribute = sensitive_attribute.squeeze()

        if self.sensitive_attribute_type == "binary":
            # Linearly scale the sensitive attribute to the initial interval
            sensitive_attribute *= (self.sensitive_attribute_interval[1] - self.sensitive_attribute_interval[0])
            sensitive_attribute += self.sensitive_attribute_interval[0]

        return sensitive_attribute

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

        model_chkpts = torch.load(self.messages_metadata[mode][round_id],
                                  map_location=torch.device(self.device))["model_state_dict"]
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
            models_dict[round_id] = self._get_model_at_round(round_id=round_id, mode=mode)

        return models_dict

    def _compute_virtual_gradient(self, global_model):
        """
        Compute the virtual gradient associated with the global model.

        Parameters:
        - global_model (torch.nn.Module): The global model for which to compute the virtual gradient.

        Returns:
        - torch.Tensor: A flattened tensor representing the virtual gradient.
        """

        global_model.zero_grad()

        grad = torch.autograd.grad(
            self.criterion(global_model(self.predicted_features), self.true_labels),
            global_model.parameters(),
            create_graph=True
        )

        grad = torch.cat([g.view(-1) for g in grad])

        return grad

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

    def _perform_iteration(self, c_iteration):
        """
         Perform one iteration of the attribute inference attack.

         Parameters:
         - c_iteration (int): The current iteration index.

         Returns:
         - Tuple of torch.Tensor: Loss and metric values for the current iteration.
         """
        self.optimizer.zero_grad()

        self.sensitive_attribute = self._sample_sensitive_attribute(deterministic=False, hard=True)
        self.predicted_features[:, self.sensitive_attribute_id] = self.sensitive_attribute

        loss = torch.tensor(0.)
        for round_id in self.round_ids:
            pseudo_grad = self.pseudo_gradients_dict[round_id]

            global_model = self.global_models_dict[round_id]
            global_model.zero_grad()

            virtual_grad = self._compute_virtual_gradient(global_model=global_model)

            # TODO: move to cosine dissimilarity
            round_loss = F.cosine_similarity(virtual_grad, pseudo_grad, dim=0)

            loss += 1 - round_loss

        self.sensitive_attribute_logits.grad = (
            torch.autograd.grad(loss, self.sensitive_attribute_logits, retain_graph=True)[0]
        )

        self.optimizer.step()

        logging.debug(f"logits: {self.sensitive_attribute_logits.squeeze()}")
        logging.debug(f"gradients: {self.sensitive_attribute_logits.grad.squeeze()}")
        logging.debug(f"true attribute: {self.true_features[:, self.sensitive_attribute_id].clone().squeeze()}")

        self.sensitive_attribute.data = self._sample_sensitive_attribute(deterministic=False, hard=True)
        self.predicted_features[:, self.sensitive_attribute_id] = self.sensitive_attribute

        metric = self.evaluate_attack()

        self.logger.add_scalar("Loss", loss, c_iteration)
        self.logger.add_scalar("Metric", metric, c_iteration)

        return loss, metric

    def execute_attack(self, num_iterations, output_losses=False):
        """
        Execute the federated learning attack on the provided dataset.

        Parameters:
        - num_iterations (int): The number of iterations to perform the attack.
        """
        # TODO: remove all_losses, only for debug
        all_losses = []
        for c_iteration in tqdm(range(num_iterations), leave=False):
            loss, metric = self._perform_iteration(c_iteration)
            all_losses.append(loss.item())

            if c_iteration % self.log_freq == 0:
                logging.info("+" * 50)
                logging.info(f"Iteration {c_iteration}: Train Loss: {loss:.4f} | Metric: {metric:.4f} ")
                logging.info("+" * 50)

        if output_losses:
            return all_losses

    def evaluate_attack(self):
        """
        Evaluate the success of the federated learning attack on the provided dataset.

        Returns:
        - torch.Tensor: The success metric value.
        """
        self.sensitive_attribute = self._sample_sensitive_attribute(deterministic=True)
        predicted_sensitive_attribute = self.sensitive_attribute.clone().detach()

        true_sensitive_attribute = self.true_features[:, self.sensitive_attribute_id].clone()

        return self.success_metric(predicted_sensitive_attribute, true_sensitive_attribute)


class ModelDrivenAttributeInferenceAttack(BaseAttributeInferenceAttack):
    def __init__(
            self, model, dataset, sensitive_attribute_id, sensitive_attribute_type, initialization, device,
            criterion, is_binary_classification, learning_rate, optimizer_name, success_metric, rng=None, torch_rng=None
    ):

        super(ModelDrivenAttributeInferenceAttack, self).__init__(
            dataset=dataset,
            sensitive_attribute_id=sensitive_attribute_id,
            sensitive_attribute_type=sensitive_attribute_type,
            initialization=initialization,
            device=device,
            criterion=criterion,
            is_binary_classification=is_binary_classification,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            success_metric=success_metric,
            rng=rng,
            torch_rng=torch_rng
        )

        self.model = model
        self.model.eval()

    def execute_attack_and_split_data(self, verbose=False):
        """
        Execute the attribute inference attack and split the data according to the attack results.

        Parameters:
        - verbose (bool): If True, activate verbose mode.
        Returns:
            df_dict: A dictionary containing the dataframes of the different splits.
        """

        df_dict = {}
        recon_error_flipped_feature = []
        recon_error_initial_feature = []
        correct_reconstructions = []
        wrong_reconstructions = []
        true_data = torch.cat((self.true_features, self.true_labels), 1).cpu().numpy()

        c_error_man = 0
        c_error_woman = 0

        c_correct_man = 0
        c_correct_woman = 0

        if self.sensitive_attribute_type == "binary":
            for idx, (_, label) in enumerate(zip(self.true_features, self.true_labels)):
                with torch.no_grad():

                    clone_1 = self.true_features[idx].clone()
                    clone_1[self.sensitive_attribute_id] = self.sensitive_attribute_interval[1]

                    clone_0 = self.true_features[idx].clone()
                    clone_0[self.sensitive_attribute_id] = self.sensitive_attribute_interval[0]

                    loss_1 = self.criterion(self.model(clone_1), label)
                    loss_0 = self.criterion(self.model(clone_0), label)

                    # TODO: correct with self.sensitive_attribute_interval
                    if loss_1 < loss_0 and self.true_features[idx, self.sensitive_attribute_id].item() < 0:
                        c_error_man += 1

                        self.predicted_features[idx, self.sensitive_attribute_id] = \
                            self.sensitive_attribute_interval[1]

                        clone_to_save_flip = self.predicted_features[idx].clone()
                        clone_to_save_flip[self.sensitive_attribute_id] = self.sensitive_attribute_interval[1]

                        clone_to_save_initial = self.predicted_features[idx].clone()
                        clone_to_save_initial[self.sensitive_attribute_id] = self.sensitive_attribute_interval[0]

                        recon_error_flipped_feature.append((torch.cat((clone_to_save_flip, label), 0).cpu().numpy()))
                        recon_error_initial_feature.append((torch.cat((clone_to_save_initial, label),
                                                                               0).cpu().numpy()))

                    elif loss_0 <= loss_1 and self.true_features[idx, self.sensitive_attribute_id].item() > 0:
                        c_error_woman += 1

                        self.predicted_features[idx, self.sensitive_attribute_id] = \
                            self.sensitive_attribute_interval[0]

                        clone_to_save_flip = self.predicted_features[idx].clone()
                        clone_to_save_flip[self.sensitive_attribute_id] = self.sensitive_attribute_interval[0]

                        clone_to_save_initial = self.predicted_features[idx].clone()
                        clone_to_save_initial[self.sensitive_attribute_id] = self.sensitive_attribute_interval[1]

                        recon_error_flipped_feature.append((torch.cat((clone_to_save_flip, label), 0).cpu().numpy()))
                        recon_error_initial_feature.append((torch.cat((clone_to_save_initial, label),
                                                                               0).cpu().numpy()))
                    else:
                        if loss_0 <= loss_1:
                            self.predicted_features[idx, self.sensitive_attribute_id] = \
                            self.sensitive_attribute_interval[0]
                            c_correct_woman += 1
                        else:
                            self.predicted_features[idx, self.sensitive_attribute_id] = \
                            self.sensitive_attribute_interval[1]
                            c_correct_man += 1

                    if torch.equal(self.true_features[idx], self.predicted_features[idx]):
                        correct_reconstructions.append((torch.cat((self.predicted_features[idx], label),
                                                                  0).cpu().numpy()))
                    else:
                        wrong_reconstructions.append((torch.cat((self.true_features[idx], label),
                                                                  0).cpu().numpy()))


        else:
            raise NotImplementedError(
                "Method 'execute_attack_and_split_data' is not yet implemented for categorical and numerical variables."
            )

        if verbose:
            logging.info(f"Number of wrong reconstructions: {c_error_woman + c_error_man}")
            logging.info(f"Number of samples incorrectly predicted as women: {c_error_woman}")
            logging.info(f"Number of samples incorrectly predicted as men: {c_error_man}")
            logging.info(f" Number of correctly classified man {c_correct_man}")
            logging.info(f" Number of correctly classified women {c_correct_woman}")
            logging.info(f"Total number of samples: {self.n_samples}")

        df_dict["recon_error_flipped_feature"] = pd.DataFrame(recon_error_flipped_feature)
        df_dict["recon_error_initial_feature"] = pd.DataFrame(recon_error_initial_feature)
        df_dict["correct_reconstructions"] = pd.DataFrame(correct_reconstructions)
        df_dict["wrong_reconstructions"] = pd.DataFrame(wrong_reconstructions)

        df_correct_recon_flip_feature = df_dict["correct_reconstructions"].copy()
        df_correct_recon_flip_feature[self.sensitive_attribute_id] = (
                1 - df_dict["correct_reconstructions"][self.sensitive_attribute_id])
        df_dict["correct_recon_flipped_feature"] = pd.DataFrame(df_correct_recon_flip_feature)

        flipped_features = true_data.copy()
        flipped_features[:, self.sensitive_attribute_id] = 1 - true_data[:, self.sensitive_attribute_id]
        df_dict["flipped_features"] = pd.DataFrame(flipped_features)

        return df_dict

    def _init_sensitive_attribute(self):
        """
        Initialize the logits of the sensitive attribute.

        Returns:
        - torch.Tensor: A tensor representing the initialized logits of the sensitive attribute with gradients enabled.
        """
        if self.initialization == "normal":
            logits = torch.randn(size=(self.num_classes,), generator=self.torch_rng)
        else:
            raise NotImplementedError(
                f"{self.initialization} is not a valid initialization strategy. "
                f"Possible are 'normal'."
            )

        logits = logits.clone().detach().requires_grad_(True).to(self.device)

        return logits

    def execute_attack(self, num_iterations, output_loss=False):
        """
        Execute the federated learning attack on the provided dataset.

        Parameters:
        - num_iterations (int): The number of iterations to perform the attack.
        - output_loss (bool): If True, the loss values for each sample are returned.
        """
        all_losses = torch.zeros(self.n_samples, 2, device=self.device)
        for idx, (_, label) in enumerate(zip(self.true_features, self.true_labels)):

            if self.sensitive_attribute_type == "binary":

                with torch.no_grad():

                    clone_1 = self.true_features[idx].clone()
                    clone_1[self.sensitive_attribute_id] = self.sensitive_attribute_interval[1]

                    clone_0 = self.true_features[idx].clone()
                    clone_0[self.sensitive_attribute_id] = self.sensitive_attribute_interval[0]

                    loss_1 = self.criterion(self.model(clone_1), label)
                    loss_0 = self.criterion(self.model(clone_0), label)

                    if loss_0 <= loss_1:
                        self.predicted_features[idx, self.sensitive_attribute_id] = \
                        self.sensitive_attribute_interval[0]
                    else:
                        self.predicted_features[idx, self.sensitive_attribute_id] = \
                        self.sensitive_attribute_interval[1]

                    all_losses[idx, 0] = loss_0
                    all_losses[idx, 1] = loss_1

            elif self.sensitive_attribute_type == "numerical":
                sensitive_attribute = self._init_sensitive_attribute()
                optimizer = self._init_optimizer(sensitive_attribute)

                for _ in range(num_iterations):

                    optimizer.zero_grad()
                    self.model.zero_grad()
                    self.predicted_features = self.predicted_features.detach()

                    self.predicted_features[idx, self.sensitive_attribute_id] = sensitive_attribute

                    loss = self.criterion(self.model(self.predicted_features[idx]), label)

                    loss.backward()
                    optimizer.step()

                self.predicted_features[idx, self.sensitive_attribute_id] = sensitive_attribute
                self.predicted_features = self.predicted_features.detach()

            elif self.sensitive_attribute_type == "categorical":
                raise NotImplementedError(
                    "Attribute Inference Attack is not yet implemented for categorical variables!"
                )
            else:
                raise ValueError(
                    f"{self.sensitive_attribute_type} is not a valid type for the sensitive attribute."
                    f"Possible are: 'binary', 'categorical', 'numerical'."
                )

        if output_loss:
            return all_losses

    def evaluate_attack(self):
        """
        Evaluate the success of the federated learning attack on the provided dataset.

        Returns:
        - torch.Tensor: The success metric value.
        """
        predicted_sensitive_attribute = self.predicted_features[:, self.sensitive_attribute_id].clone()

        true_sensitive_attribute = self.true_features[:, self.sensitive_attribute_id].clone()

        return self.success_metric(predicted_sensitive_attribute, true_sensitive_attribute)
