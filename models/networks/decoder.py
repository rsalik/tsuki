import torch
import torch.nn as nn
from ..blocks.residual import ResidualBlock


class Upsampler(nn.Module):
    """Upsampler that reverses the Downsampler operations."""

    def __init__(self, in_channels, out_channels, target_h, target_w):
        """
        :param in_channels: int, input channels (num_channels from latent)
        :param out_channels: int, output channels (C from observation)
        :param target_h: int, target height of output
        :param target_w: int, target width of output
        """
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w

        # Reverse of pooling2 + resblocks3
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(in_channels, in_channels) for _ in range(1)]
        )

        # Reverse of pooling1 + resblocks2
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(in_channels, in_channels) for _ in range(1)]
        )

        # Reverse of downsample_block (stride 2 conv -> transposed conv)
        self.upsample_block = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn_up = nn.BatchNorm2d(in_channels // 2)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(in_channels // 2, in_channels // 2) for _ in range(1)]
        )

        # Reverse of conv1 (stride 2 -> transposed conv)
        self.conv_final = nn.ConvTranspose2d(
            in_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x):
        # Upsample path (reverse of downsampler)
        x = self.upsample1(x)
        for block in self.resblocks1:
            x = block(x)

        x = self.upsample2(x)
        for block in self.resblocks2:
            x = block(x)

        x = self.upsample_block(x)
        x = self.bn_up(x)
        x = nn.functional.relu(x)
        for block in self.resblocks3:
            x = block(x)

        x = self.conv_final(x)

        # Ensure output matches target size exactly
        x = nn.functional.interpolate(
            x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
        )

        return x


class DecoderNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        upsample=True,
    ):
        """
        Decoder network - reconstructs observations from latent states.

        :param observation_shape: tuple or list, shape of observations: [C, H, W]
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states (input latent channels)
        :param upsample: bool, True -> do upsampling to match observation shape
        """
        super().__init__()
        self.upsample = upsample
        self.num_channels = num_channels
        self.observation_shape = observation_shape

        # Residual blocks to process latent
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

        # Upsampling layers
        if self.upsample:
            self.upsample_net = Upsampler(
                num_channels,
                observation_shape[0],  # Output channels = observation channels
                observation_shape[1],  # Target H
                observation_shape[2],  # Target W
            )
        else:
            # Simple projection without upsampling
            self.conv = nn.Conv2d(
                num_channels,
                observation_shape[0],
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(observation_shape[0])

    def forward(self, x):
        """
        Forward pass for single latent state.

        Args:
            x: Input tensor of shape [batch, num_channels, H', W']

        Returns:
            Output tensor of shape [batch, C, H, W] matching observation_shape
        """
        # Apply residual blocks
        for block in self.resblocks:
            x = block(x)

        # Upsample to observation shape
        if self.upsample:
            x = self.upsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        return x
