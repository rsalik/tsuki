from ..base import Network
import torch.nn as nn


def atari_model(in_channels, out_channels):
    DownSample = Network(
        [
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 2),
        ]
    )
