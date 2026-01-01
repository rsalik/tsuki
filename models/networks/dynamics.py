import torch
import torch.nn as nn
from ..blocks.residual import ResidualBlock, conv3x3

from flash_stu import FlashSTUBlock


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        action_space_size,
        spatial_size=6,
        sequence_length=16,
        is_continuous=False,
        action_embedding=False,
        action_embedding_dim=32,
    ):
        """
        Dynamics network with STU for spectral filtering over latent state sequences.

        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        :param spatial_size: int, spatial dimension of latent state (H' = W')
        :param sequence_length: int, number of timesteps for STU
        :param is_continuous: bool, whether action space is continuous
        :param action_embedding: bool, whether to use action embedding
        :param action_embedding_dim: int, dimension of action embedding
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.spatial_size = spatial_size
        self.sequence_length = sequence_length

        # Action encoding layers
        if action_embedding:
            self.conv1x1 = nn.Conv2d(
                action_space_size if is_continuous else 1, self.action_embedding_dim, 1
            )
            self.ln = nn.LayerNorm([action_embedding_dim, spatial_size, spatial_size])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(
                num_channels + action_space_size if is_continuous else num_channels + 1,
                num_channels,
            )

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

        # STU for spectral filtering over latent state sequences
        # Flattened spatial dimension for STU input
        self.spatial_dim = num_channels * spatial_size * spatial_size

        # Projection layers for STU
        self.spatial_to_stu = nn.Linear(self.spatial_dim, num_channels)

        self.stu_block = FlashSTUBlock(
            d_model=num_channels,
            sequence_length=sequence_length,
            num_filters=24,
            use_attention=False,  # Pure STU, no attention
        )

        self.stu_to_spatial = nn.Linear(num_channels, self.spatial_dim)

    def forward(self, states, actions):
        """
        Forward pass for sequence of states and actions.

        Args:
            states: Input tensor of shape [batch, timesteps, num_channels, H', W']
            actions: Action tensor of shape [batch, timesteps, action_dim] or [batch, timesteps, 1]

        Returns:
            Output tensor of shape [batch, timesteps, num_channels, H', W']
        """
        batch_size, timesteps, channels, h, w = states.shape

        # 1. First apply STU over the input state sequence for spectral filtering
        # Flatten spatial dims: [batch, timesteps, C, H, W] -> [batch, timesteps, C*H*W]
        x = states.view(batch_size, timesteps, channels * h * w)

        # Project to STU dimension and apply STU
        x = self.spatial_to_stu(x)
        x = self.stu_block(x)
        x = self.stu_to_spatial(x)

        # Reshape back: [batch, timesteps, spatial_dim] -> [batch, timesteps, C, H, W]
        x = x.view(batch_size, timesteps, channels, h, w)

        # 2. Now process each timestep with action conditioning
        # Merge batch and time: [batch, timesteps, C, H, W] -> [batch * timesteps, C, H, W]
        state = x.view(batch_size * timesteps, channels, h, w)

        # Reshape actions for processing
        if not self.is_continuous:
            # actions shape: [batch, timesteps, 1] -> [batch * timesteps, 1]
            action = actions.view(batch_size * timesteps, -1)
            action_place = torch.ones(
                (batch_size * timesteps, 1, h, w), device=state.device
            ).float()
            action_place = (
                action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action = actions.view(batch_size * timesteps, -1)
            action_place = action.reshape(batch_size * timesteps, -1, 1, 1).repeat(
                1, 1, h, w
            )

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        # Combine state and action
        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        # Unmerge batch and time: [batch * timesteps, C, H, W] -> [batch, timesteps, C, H, W]
        x = x.view(batch_size, timesteps, channels, h, w)

        return x
