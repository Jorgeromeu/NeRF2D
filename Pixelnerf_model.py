import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding

class SimpleNeRF(nn.Module):
    """
    Simplified NeRF architecture with no view dir or skip connections
    """

    def __init__(
            self,
            d_input: int = 2,
            d_emb: int = 8,
            n_layers: int = 8,
            d_hidden: int = 256,
    ):
        super().__init__()

        self.d_input = d_input

        # positional encoding
        self.pe = PositionalEncoding(d_emb)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.pe.out_dim(d_input), d_hidden)] +
            [nn.Linear(d_hidden, d_hidden) for i in range(n_layers - 1)] +
            [nn.Linear(d_hidden, 4)]
        )

    def forward(
            self,
            x: torch.Tensor,
            d: torch.Tensor
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

class NeRF(nn.Module):
    """
    Simplified NeRF architecture with no view dir or skip connections
    """

    def __init__(
            self,
            d_pos_input: int = 2,
            d_dir_input: int = 1,
            n_freqs_position: int = 10,
            n_freqs_direction: int = 4,
            skip_indices: list[int] = None,
            n_layers: int = 8,
            d_hidden: int = 256,
            if_hidden: int = 1000
    ):
        super().__init__()

        if skip_indices is None:
            self.skip_indices = []
        else:
            self.skip_indices = skip_indices

        self.d_input = d_pos_input

        # positional encoding
        self.pos_pe = PositionalEncoding(n_freqs_position)
        self.dir_pe = PositionalEncoding(n_freqs_direction)

        # size of the inputs after positional encoding
        d_x_enc = self.pos_pe.out_dim(d_pos_input)
        d_d_enc = self.dir_pe.out_dim(d_dir_input)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(d_x_enc + if_hidden + d_d_enc, d_hidden)] +
            [nn.Linear(d_hidden + d_x_enc, d_hidden) if i + 1 in self.skip_indices else
             nn.Linear(d_hidden, d_hidden)
             for i in range(n_layers - 1)]
        )

        # maps to density
        self.density_head = nn.Linear(d_hidden, 1)

        # maps to color, with viewdir
        self.color_layer = nn.Linear(d_hidden, d_hidden // 2)
        self.color_head = nn.Linear(d_hidden // 2, 3)

    def forward(
            self,
            x: torch.Tensor,
            d: torch.Tensor,
            image_features: torch.Tensor
    ) -> torch.Tensor:
        # apply positional encoding
        pos_enc = self.pos_pe(x)
        dir_enc = self.dir_pe(d)

        # apply model layers
        z = torch.cat([pos_enc, image_features, dir_enc], dim = 1)
        for i, layer in enumerate(self.layers):

            if i not in self.skip_indices:

                z = self.relu(layer(z))

            else:
                z = torch.cat([z, pos_enc], dim=1)
                z = self.relu(layer(z))

        # immediately get density
        density = self.relu(self.density_head(z))

        # concatenate feature map with view direction
        z = torch.cat([z], dim=1)
        z = self.relu(self.color_layer(z))
        color = self.sigmoid(self.color_head(z))

        output = torch.cat([color, density], dim=1)

        return output
