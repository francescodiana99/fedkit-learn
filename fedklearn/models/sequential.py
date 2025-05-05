import torch.nn as nn


class SequentialNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_layers=None):
        super(SequentialNet, self).__init__()

        if hidden_layers is None:
            hidden_layers = []

        if len(hidden_layers) > 0:
            self.layers = nn.ModuleList([
                nn.Linear(input_dimension, hidden_layers[0]),
                nn.ReLU()
            ])

            for i in range(1, len(hidden_layers)):
                self.layers.extend([
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    nn.ReLU()
                ])

            self.layers.append(nn.Linear(hidden_layers[-1], output_dimension))

        else:
            self.layers = nn.ModuleList([nn.Linear(input_dimension, output_dimension)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x