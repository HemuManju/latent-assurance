from torch import nn
import pytorch_lightning as pl
import torch


from src.visualization.visualize import interactive_show_grid

from .utils import build_model, Flatten


class ConvNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet, self).__init__()

        # Parameters
        obs_size = hparams['obs_size']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, obs_size, 64, 64))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 64, 64)
            nn.Conv2d(obs_size, 16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        x = self.cnn_base(x)
        x = torch.flatten(x, start_dim=1)
        q_values = self.fc(x)
        return q_values
