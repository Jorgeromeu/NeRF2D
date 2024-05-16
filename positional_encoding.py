import torch

from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_emb):

        super().__init__()
        self.d_emb = d_emb

        # compute frequency banngds
        self.frequency_bands = 2.0 ** torch.linspace(
            0.0,
            d_emb - 1,
            d_emb,
        )

    def out_dim(self, in_coords: int):
        return in_coords * 2 * self.d_emb

    def forward(self, x):
        encoding = []
        for freq in self.frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x * freq))

        return torch.cat(encoding, dim=-1)
