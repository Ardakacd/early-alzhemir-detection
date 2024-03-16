import torch
import torch.nn as nn

class ArtificialNeuralNetwork(nn.Module):                           ## nn.Module is super class of our class
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ArtificialNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):                           # implementation of forward prop
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)       # No activation function applied to the output layer