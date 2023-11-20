import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, output_dimension, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = output_dimension
        self.fc = nn.Linear(input_dimension, output_dimension, bias=bias)

    def forward(self, x):
        return self.fc(x)


class TwoLinearLayers(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, bias=True):
        super(TwoLinearLayers, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dim = output_dimension

        self.fc1 = nn.Linear(self.input_dimension, self.hidden_dimension, bias=bias)
        self.fc2 = nn.Linear(self.hidden_dimension, self.output_dim, bias=bias)

    def forward(self, x):
        return self.fc2(self.fc1(x))
