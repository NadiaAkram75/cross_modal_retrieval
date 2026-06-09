import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 32

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.embedding = nn.Linear(32, self.embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding(x)

        return x

class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()


        self.relu = nn.ReLU()

        self.fc = nn.Linear(32, 32)


        self.deconv1 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )

        self.deconv2 = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2
        )


        self.deconv3 = nn.ConvTranspose3d(
            in_channels=8,
            out_channels=1,
            kernel_size=2,
            stride=2
        )

    def forward(self, z):
        x = self.fc(z)

        x = x.view(-1, 32, 1, 1, 1)

        x = self.deconv1(x)
        x = self.relu(x)


        x = self.deconv2(x)
        x = self.relu(x)

        x = self.deconv3(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction, z