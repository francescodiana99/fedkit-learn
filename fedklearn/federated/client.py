class Client:
    """
    A class representing a federated learning client.

    This class encapsulates the functionality of a federated learning client, including
    training and testing operations, as well as logging metrics.

    Attributes:
    - trainer: The trainer responsible for training the model on the client.
    - train_loader: DataLoader for the training dataset.
    - test_loader: DataLoader for the testing dataset.
    - local_steps: Number of local training steps or epochs.
    - by_epoch: Boolean indicating whether training is performed by epoch or by step.
    - logger: The logger for recording client-specific logs.
    - n_train_samples: Number of samples in the training dataset.
    - n_test_samples: Number of samples in the testing dataset.
    - counter: Counter to track the number of training steps or epochs.

    - _train_iterator: Iterator for the training DataLoader.

    Methods:
    - get_next_batch: Retrieve the next batch from the training DataLoader.
    - step: Perform a training step or epoch based on the specified configuration.
    - write_logs: Log training and testing metrics using the provided logger.
    """

    def __init__(self, trainer, train_loader, test_loader, local_steps, by_epoch, logger, name=None):
        """
        Initialize a federated learning client.

        Parameters:
        - trainer: The trainer responsible for training the model on the client.
        - train_loader: DataLoader for the training dataset.
        - test_loader: DataLoader for the testing dataset.
        - local_steps: Number of local training steps or epochs.
        - by_epoch: Boolean indicating whether training is performed by epoch or by step.
        - logger: The logger for recording client-specific logs.
        - name (str, Optional): client name
        """
        self.trainer = trainer

        self.train_loader = train_loader
        self.test_loader = test_loader

        self._train_iterator = iter(self.train_loader)

        self.local_steps = local_steps
        self.by_epoch = by_epoch

        self.logger = logger

        self.name = name

        self.n_train_samples = len(self.train_loader.dataset)
        self.n_test_samples = len(self.test_loader.dataset)

        self.counter = 0

    def get_next_batch(self):
        """
        Retrieve the next batch from the training DataLoader.

        Returns:
        - batch: The next batch of data.
        """
        try:
            batch = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            batch = next(self._train_iterator)

        return batch

    def step(self):
        """
         Perform training steps or epochs based on the specified configuration.
         """
        self.counter += 1

        if self.by_epoch:
            self.trainer.fit_epochs(loader=self.train_loader, n_epochs=self.local_steps)

        else:
            for _ in range(self.local_steps):
                c_batch = self.get_next_batch()
                self.trainer.fit_batch(c_batch)

    def update_trainer(self, trainer):
        self.trainer.update(trainer)

    def write_logs(self):
        """
        Log training and testing metrics using the provided logger.
        """
        train_loss, train_metric = self.trainer.evaluate_loader(loader=self.train_loader)
        test_loss, test_metric = self.trainer.evaluate_loader(loader=self.test_loader)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_metric, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_metric, self.counter)

        return train_loss, train_metric, test_loss, test_metric
