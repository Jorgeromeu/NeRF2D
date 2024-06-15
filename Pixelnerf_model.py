import torch
import torch.nn as nn
from einops import rearrange

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
            if_hidden: int = 120,
            nr_images: int = 5
    ):
        super().__init__()

        if skip_indices is None:
            self.skip_indices = []
        else:
            self.skip_indices = skip_indices

        self.d_input = d_pos_input

        self.nr_images = nr_images
        # positional encoding
        self.pos_pe = PositionalEncoding(n_freqs_position)
        # self.dir_pe = PositionalEncoding(n_freqs_direction)

        # size of the inputs after positional encoding
        d_x_enc = self.pos_pe.out_dim(d_pos_input)
        # d_d_enc = self.dir_pe.out_dim(d_dir_input + d_dir_input)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool1d(nr_images)

        self.feature_layer = nn.Linear(if_hidden, d_hidden)
        self.query_layers = [nn.Linear(d_x_enc + 2, d_hidden) for i in range(self.nr_images)]

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(d_hidden + d_hidden, d_hidden)] +
            [nn.Linear(d_hidden * 3, d_hidden) if i + 1 in self.skip_indices else
             nn.Linear(d_hidden, d_hidden)
             for i in range(n_layers - 1)])


        # maps to color, with viewdir
        self.final_layer = nn.Linear(d_hidden, d_hidden // 2)
        self.final_head = nn.Linear(d_hidden // 2, 4)

    def forward(
            self,
            x,
            d,
            image_features
    ) -> torch.Tensor:
        # apply positional encoding
        # dir_enc = self.dir_pe(d)

        # apply model layers

        v = []

        for i in range(len(image_features)):
            pos_enc = self.pos_pe(x[i])
            cur_dir = d[i]
            # dir_enc = self.dir_pe(d[i])

            cur_image_features = rearrange(image_features[i], 'a b -> b a')
            # check_image_features = np.numpy(cur_image_features)
            features = self.relu(self.feature_layer(cur_image_features))
            # 24 + 12

            z = torch.cat([pos_enc, cur_dir], dim = 1)

            query = self.relu(self.query_layers[i](z))

            z = torch.cat([features, query], dim=1)

            j = z

            for i, layer in enumerate(self.layers):
                if i not in self.skip_indices:

                    z = self.relu(layer(z))

                else:
                    z = torch.cat([j, z], dim=1)
                    z = self.relu(layer(z))
            v.append(z)
            # immediately get density

        stacked_v = torch.stack(v, dim = 0)
        stacked_v = stacked_v.permute(1, 2, 0)
        z = self.avgpool(stacked_v).squeeze()



        # concatenate feature map with view direction
        z = self.relu(self.final_layer(z))


        output = self.sigmoid(self.final_head(z))

        return output
