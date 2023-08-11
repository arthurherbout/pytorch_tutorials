import torch
from torch import nn


class FashionMnistNetwork(nn.Module):
    """
    This class defines the neural network used to classify images in the FashionMNIST dataset
    """
    def __init__(self, n_hidden_features: int = 512):
        """

        Args:
            n_hidden_features: number of features in the hidden layers of the neural network.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, n_hidden_features),
            nn.ReLU(),
            nn.Linear(n_hidden_features, n_hidden_features),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass on the network.
        Args
            x: batch of images to classify
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

