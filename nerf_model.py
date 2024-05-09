import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding

class SimpleNeRF(nn.Module):
    """
    Simplified NeRF architecture with no view dir or skip connections
    """

    def __init__(
            self,
            d_input: int = 3,
            n_layers: int = 8,
            d_hidden: int = 256,
    ):
        super().__init__()

        self.d_input = d_input

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # positional encoding
        self.pe = PositionalEncoding(d_input)

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.pe.out_dim(d_input), d_hidden)] +
            [nn.Linear(d_hidden, d_hidden) for i in range(n_layers - 1)] +
            [nn.Linear(d_hidden, 4)]
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        # apply positional encoding
        x = self.pe(x)

        # apply model layers
        for layer in self.layers:
            x = self.relu(layer(x))

        color = self.sigmoid(x[:, :3])
        density = self.relu(x[:, 3:])
        output = torch.cat([color, density], dim=1)

        return output
