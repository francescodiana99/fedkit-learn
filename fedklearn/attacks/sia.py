import torch


class SourceInferenceAttack:
    """
    Class representing a source inference attack in federated learning.
    """
    def __init__(self, attacked_client_id, dataloader, trainers_dict):
        self.dataloader = dataloader
        self.trainers_dict = trainers_dict

        self.num_clients = len(self.trainers_dict)

        self.attacked_client_id = int(attacked_client_id)

        self.losses = [[] for _ in range(self.num_clients)]

    def execute_attack(self):
        """

        Returns:
            None
        """
        for batch in self.dataloader:
            for client_id in self.trainers_dict:
                trainer = self.trainers_dict[client_id]

                loss = trainer.compute_loss(batch)

                self.losses[int(client_id)].append(loss)

    def evaluate_attack(self):
        """

        Returns:
            None
        """
        losses = torch.hstack([torch.cat(self.losses[client_id]) for client_id in range(self.num_clients)])

        score = (losses.argmin(axis=1) == self.attacked_client_id).sum() / losses.shape[0]

        return float(score)
