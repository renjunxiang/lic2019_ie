import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = x.transpose(2, 1)
        x_conv = self.conv1(x)
        x_conv = nn.ReLU()(x_conv)
        x_conv = self.conv2(x_conv)
        x = x + x_conv
        x = nn.ReLU()(x)

        return x.transpose(2, 1)
