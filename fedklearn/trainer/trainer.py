import logging

import torch

from ..utils import *

class Trainer:
    """
    Responsible for training and evaluating a (deep-)learning model.

    Attributes
    ----------
    model (nn.Module): The model trained by the Trainer.
    model_name (str): Optional name for the model.
    criterion (torch.nn.modules.loss): Loss function used to train the `model`. Should have reduction="none".
    metric (callable): Function to compute the metric. Should accept two vectors and return a scalar.
    device (str or torch.Device): Device on which to perform computations.
    optimizer (torch.optim.Optimizer): Optimization algorithm.
    lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    is_binary_classification (bool): Whether to cast labels to float or not. Set to True if using BCELoss.

    Methods
    -------
    update(trainer):
        Update the trainer's model by copying the state dictionary from a given trainer's model.

    update_model(model):
        Update the trainer's model with the parameters from the provided model.

    compute_stochastic_gradient(batch):
        Compute the stochastic gradient on one batch, the result is directly stored in `self.model.parameters()`.

    compute_loss(batch):
        Compute the loss on one batch.

    fit_batch(batch):
        Perform an optimizer step over one batch.

    fit_epoch(loader):
        Perform several optimizer steps on all batches drawn from `loader`.

    evaluate_loader(loader):
        Evaluate Trainer on `loader`.

    fit_epochs(loader, n_epochs):
        Perform multiple training epochs.

    get_param_tensor():
        Get `model` parameters as a unique flattened tensor.

    set_param_tensor(param_tensor):
        Set the parameters of the model from `param_tensor`.

    get_grad_tensor():
        Get `model` gradients as a unique flattened tensor.

    set_grad_tensor(grad_tensor):
        Set the gradients of the model from `grad_tensor`.

    save_checkpoint(path):
        Save the model, the optimizer, and the learning rate scheduler state dictionaries.

    load_checkpoint(path):
        Load the model, the optimizer, and the learning rate scheduler state dictionaries.

    freeze_model():
        Freeze the model by setting the `requires_grad` attribute to `False` for all parameters.

    """
    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer,
            model_name=None,
            lr_scheduler=None,
            is_binary_classification=False,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.is_binary_classification = is_binary_classification

        self.n_modules = self.__get_num_modules()
        self.model_dim = int(self.get_param_tensor().shape[0])

    def __get_num_modules(self):
        """
        Computes the number of modules in the model network.

        Returns
        -------
        n_modules (int): The size of `self.model.modules()`.
        """
        n_modules = 0
        for _ in self.model.modules():
            n_modules += 1

        return n_modules

    def update(self, trainer):
        """
        Update the trainer's model by copying the state dictionary from a given trainer's model.

        Parameters:
        - trainer (Trainer): The trainer whose model's state will be copied to update the client's model.
        """
        copy_model(target=self.model, source=trainer.model)

    def update_model(self, model):
        """
        Update the trainer's model with the parameters from the provided model.

        This method copies the state dictionary from the provided model to the trainer's model,
        effectively updating the trainer's model with the parameters of the provided model.

        Parameters:
        - model: PyTorch model containing the parameters to update the trainer's model.
        """
        copy_model(target=self.model, source=model)

    def compute_stochastic_gradient(self, batch):
        """
        Compute the stochastic gradient on one batch.

        Parameters
        ----------
        batch : tuple
            A tuple (x, y) representing input features (x) and corresponding labels (y).

        Returns
        -------
        None
            The method updates the model's parameters in-place.

        Notes
        -----
        This method performs the following steps:
        1. Transfers input features (x) and labels (y) to the device specified during Trainer initialization.
        2. If binary classification is enabled, casts labels (y) to float and adds a singleton dimension.
        3. Resets the gradients of the model's parameters.
        4. Computes the predicted output (y_pred) using the model.
        5. Computes the loss between the predicted output and the true labels.
        6. Computes gradients with respect to the loss.
        """
        self.model.train()

        x, y = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32)

        self.optimizer.zero_grad()

        y_pred = self.model(x)

        y_pred = y_pred.squeeze()

        loss = self.criterion(y_pred, y)

        loss.backward()

    def compute_loss(self, batch):
        """
        Compute the loss on one batch.

        Parameters
        ----------
        batch : tuple
            A tuple (x, y) representing input features (x) and corresponding labels (y).

        Returns
        -------
        torch.Tensor
            the computed loss on the batch

        Notes
        -----
        This method performs the following steps:
        1. Transfers input features (x) and labels (y) to the device specified during Trainer initialization.
        2. If binary classification is enabled, casts labels (y) to float and adds a singleton dimension.
        3. Resets the gradients of the model's parameters.
        4. Computes the predicted output (y_pred) using the model.
        5. Computes the loss between the predicted output and the true labels.
        6. Computes gradients with respect to the loss.
        """
        self.model.eval()

        with torch.no_grad():
            x, y = batch
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32)

            y_pred = self.model(x)

            if y_pred.shape[0] != 1:
                y_pred = y_pred.squeeze()
            else:
                y_pred = y_pred.squeeze(dim=tuple(y_pred.shape[1:]))


            loss = self.criterion(y_pred, y)

        return loss

    def fit_batch(self, batch):
        """
        Perform an optimizer step over one batch.

        Parameters
        ----------
        batch : tuple
            A tuple (x, y) representing input features (x) and corresponding labels (y).

        Returns
        -------
        tuple
            A tuple (loss, metric), where loss is the value of the loss function and
            metric is the computed evaluation metric.

        Notes
        -----
        This method performs the following steps:
        1. Sets the model to training mode.
        2. Transfers input features (x) and labels (y) to the device specified during Trainer initialization.
        3. If binary classification is enabled, casts labels (y) to float and adds a singleton dimension.
        4. Resets the gradients of the model's parameters.
        5. Computes the predicted output (y_pred) using the model.
        6. Computes the loss between the predicted output and the true labels.
        7. Computes the evaluation metric, normalized by the length of the labels.
        8. Computes gradients with respect to the loss.
        9. Performs an optimizer step to update the model's parameters.
        10. If a learning rate scheduler is provided, takes a step in its schedule.
        """

        self.model.train()

        x, y = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32)

        self.optimizer.zero_grad()

        y_pred = self.model(x)

        y_pred = y_pred.squeeze()

        loss = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y)

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.item(), metric

    def fit_epoch(self, loader):
        """
        Perform multiple optimizer steps on all batches drawn from the provided data loader.

        Parameters
        ----------
        loader (torch.utils.data.DataLoader): DataLoader providing batches of input features and corresponding labels.

        Returns
        -------
        tuple
            A tuple (average_loss, average_metric), where average_loss is the average value of the loss
            function over all batches, and average_metric is the average computed evaluation metric.

        Notes
        -----
        This method performs the following steps:
        1. Sets the model to training mode.
        2. Initializes global loss, global metric, and the total number of samples.
        3. Iterates over batches from the loader.
        4. Transfers input features (x) and labels (y) to the device specified during Trainer initialization.
        5. If binary classification is enabled, casts labels (y) to float and adds a singleton dimension.
        6. Resets the gradients of the model's parameters.
        7. Computes the predicted output (y_pred) using the model.
        8. Computes the loss between the predicted output and the true labels.
        9. Computes gradients with respect to the loss.
        10. Performs an optimizer step to update the model's parameters.
        11. Updates global_loss and global_metric.
        12. Computes average loss and average metric over all batches.
        """
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            # condition needed to avoid squeezing the batch size if it is 1
            if y_pred.shape[0] != 1:
                y_pred = y_pred.squeeze()
            else:
                y_pred = y_pred.squeeze(dim=tuple(y_pred.shape[1:]))

            loss = self.criterion(y_pred, y)

            loss.backward()

            self.optimizer.step()

            global_loss += loss.item() * y.size(0)
            global_metric += self.metric(y_pred, y) * y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def evaluate_loader(self, loader, output_losses=False):
        """
        Evaluate the Trainer on a provided data loader.

        Parameters
        ----------
        loader (torch.utils.data.DataLoader): DataLoader providing batches of input features
            and corresponding labels for evaluation.

        Returns
        -------
        tuple
            A tuple (average_loss, average_metric) representing the average loss and average evaluation metric
            accumulated over all batches in the loader.

        Notes
        -----
        This method performs the following steps:
        1. Sets the model to evaluation mode.
        2. Initializes global loss, global metric, and the total number of samples.
        3. Uses torch.no_grad() to disable gradient computation during evaluation.
        4. Iterates over batches from the loader.
        5. Transfers input features (x) and labels (y) to the device specified during Trainer initialization.
        6. If binary classification is enabled, casts labels (y) to float and adds a singleton dimension.
        7. Computes the predicted output (y_pred) using the model.
        8. Updates global_loss and global_metric.
        9. Computes average loss and average metric over all batches.
        """

        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32)

                y_pred = self.model(x)

                if y_pred.shape[0] != 1:
                    y_pred = y_pred.squeeze()
                else:
                    y_pred = y_pred.squeeze(dim=tuple(y_pred.shape[1:]))

                global_loss += self.criterion(y_pred, y).item() * y.size(0)
                global_metric += self.metric(y_pred, y) * y.size(0)

                n_samples += y.size(0)

            return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, loader, n_epochs):
        """
        Perform multiple training epochs.

        Parameters
        ----------
        loader (torch.utils.data.DataLoader): DataLoader providing batches
            of input features and corresponding labels for training.
        n_epochs (int): Number of successive epochs (passes through the entire dataset).

        Returns
        -------
        None

        Notes
        -----
        This method iterates over the specified number of epochs, performing the following steps in each epoch:
        1. Calls the `fit_epoch` method to perform optimizer steps on all batches from the loader.
        2. If a learning rate scheduler is provided (not None), calls its `step` method to update the learning rate.
        """

        for step in range(n_epochs):
            self.fit_epoch(loader)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_param_tensor(self):
        """
        Get the parameters of the Trainer's model as a unique flattened tensor.

        Returns
        -------
        torch.Tensor: A flattened tensor containing the concatenated parameters of the model.

        Notes
        -----
        This method iterates over all parameters of the Trainer's model and flattens them into a single tensor.
        """
        return get_param_tensor(self.model)

    def set_param_tensor(self, param_tensor):
        """
        Set the parameters of the Trainer's model from the provided tensor.

        Parameters
        ----------
        param_tensor (torch.Tensor): Tensor of shape (`self.model_dim`) containing the parameters to
            update the Trainer's model.

        Notes
        -----
        This method iterates over all parameters of the Trainer's model, extracts the corresponding portion
        from the provided tensor, and reshapes it to match the original parameter shape.
        """
        set_param_tensor(model=self.model, param_tensor=param_tensor, device=self.device)

    def get_grad_tensor(self):
        """
        Get `model` gradients as a unique flattened tensor.

        Returns
        -------
        torch.Tensor
            Flattened tensor containing gradients of the Trainer's model parameters.
        """

        return get_grad_tensor(model=self.model)

    def set_grad_tensor(self, grad_tensor):
        """
        Set the gradients of the Trainer's model from the provided tensor.

        Parameters
        ----------
        grad_tensor (torch.Tensor): Tensor of shape (`self.model_dim`) containing gradients to update
            the Trainer's model.

        Notes
        -----
        This method iterates over all parameters of the Trainer's model, extracts the corresponding portion
        from the provided tensor, and reshapes it to match the original parameter shape.
        """
        set_grad_tensor(model=self.model, grad_tensor=grad_tensor, device=self.device)

    def save_checkpoint(self, path):
        """
        Save the Trainer's model, optimizer, and learning rate scheduler state dictionaries.

        Parameters
        ----------
        path : str
            The path to a .pt file to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load the Trainer's model, optimizer, and learning rate scheduler state dictionaries from a checkpoint.

        Parameters
        ----------
        path : str
            The path to a .pt file storing the required data.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def freeze_model(self):
        """
        Freeze the model by setting the `requires_grad` attribute to `False` for all parameters.

        Returns:
            None
        """
        for param in self.model.parameters():
            param.require_grad = False

        self.model.eval()

class DebugTrainer(Trainer):
    """Trainer with additional debugging features."""
    def __init__(self, model,criterion,metric,device,optimizer, model_name=None,lr_scheduler=None,
                 is_binary_classification=False):
        super(DebugTrainer, self).__init__(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            model_name=model_name,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
        self.x_grad = None

    def get_grad_by_layer(self):
            for name, param in self.model.named_parameters():
                print(f"Layer: {name}")
                print(param.grad)
                print(torch.nonzero(param.grad))
                print("")

    def fit_epoch(self, loader):
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y in loader:
            x = x.to(self.device).type(torch.float32)
            x.requires_grad = True
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            if y_pred.shape[0] != 1:
                y_pred = y_pred.squeeze()
            else:
                y_pred = y_pred.squeeze(dim=tuple(y_pred.shape[1:]))

            loss = self.criterion(y_pred, y)

            loss.backward()

            self.x_grad = x.grad

            self.optimizer.step()

            global_loss += loss.item() * y.size(0)
            global_metric += self.metric(y_pred, y) * y.size(0)

        return global_loss / n_samples, global_metric / n_samples
    def fit_epochs_check_gradient(self, loader, n_epochs, n_debug_epoch):
        for step in range(n_epochs):
            self.fit_epoch(loader)
            if step % n_debug_epoch == 0:
                self.get_grad_by_layer()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def fit_epochs(self, loader, n_epochs):
        for step in range(n_epochs):
            print(self.fit_epoch(loader))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
