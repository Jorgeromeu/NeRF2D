import torch
import torch.nn as nn



"""
Basic, extremely simple convolutional encoder
"""

import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        torch.nn.init._no_grad_normal_(self.conv1.weight, 0, 1)

        self.bn1 = nn.BatchNorm1d(500)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # Defining layers (layer1, layer2, layer3) as examples
        self.layer1 = self._make_layer(10, 20, 2, 500)
        self.layer2 = self._make_layer(20, 40, 2, 500)
        self.layer3 = self._make_layer(40, 50, 2, 500)

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False

        for param in self.relu.parameters():
            param.requires_grad = False
        for param in self.maxpool.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # if self.latent_size != 512:
        #     self.fc = nn.Linear(512, self.latent_size)

        # self.latent = None

    def _make_layer(self, in_channels, out_channels, blocks, nr_batch):
        layers = []
        for _ in range(blocks):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            torch.nn.init._no_grad_normal_(self.conv1.weight, 0, 1)
            layers.append(conv1d)
            layers.append(nn.BatchNorm1d(nr_batch))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x

