import torch
import torch.nn as nn
from .downsampler import Downsampler
from ..blocks.residual import ResidualBlock


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        downsample,
    ):
        """
        Encoder network for single frames.

        :param observation_shape: tuple or list, shape of observations: [C, H, W]
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param downsample: bool, True -> do downsampling for observations
        """
        super().__init__()
        self.downsample = downsample
        self.num_channels = num_channels

        # Spatial processing layers
        if self.downsample:
            self.downsample_net = Downsampler(
                observation_shape[0],
                num_channels,
            )
        else:
            self.conv = conv3x3(
                observation_shape[0],
                num_channels,
            )
            self.bn = nn.BatchNorm2d(num_channels)

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        """
        Forward pass for single frame.

        Args:
            x: Input tensor of shape [batch, C, H, W]

        Returns:
            Output tensor of shape [batch, num_channels, H', W']
        """
        # Apply spatial processing (downsampler/conv + resblocks)
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        return x
