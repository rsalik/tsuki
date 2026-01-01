"""
World Model combining Encoder, Dynamics (with STU), and Decoder.
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """
    World Model for Atari next-frame prediction.

    Architecture:
        Encoder: frame -> latent
        Dynamics + STU: (latent sequence, actions) -> predicted next latents
        Decoder: latent -> reconstructed frame
        Reward Head: latent -> predicted reward
    """

    def __init__(
        self,
        observation_shape=(4, 84, 84),  # [C, H, W] - stacked frames
        num_channels=64,
        num_blocks=2,
        action_space_size=18,
        sequence_length=16,
        spatial_size=6,  # After downsampling: 84 // 16 ≈ 5-6
    ):
        """
        Args:
            observation_shape: Shape of observations [C, H, W]
            num_channels: Hidden dimension for latent states
            num_blocks: Number of residual blocks in each network
            action_space_size: Number of discrete actions
            sequence_length: Number of timesteps for STU
            spatial_size: Spatial dimension of latent (H/16, W/16)
        """
        super().__init__()

        self.observation_shape = observation_shape
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.spatial_size = spatial_size

        # Import here to avoid circular imports
        from models.networks.encoder import EncoderNetwork
        from models.networks.dynamics import DynamicsNetwork
        from models.networks.decoder import DecoderNetwork

        # Encoder: frame -> latent
        self.encoder = EncoderNetwork(
            observation_shape=observation_shape,
            num_blocks=num_blocks,
            num_channels=num_channels,
            downsample=True,
        )

        # Dynamics + STU: (latent sequence, actions) -> predicted latents
        self.dynamics = DynamicsNetwork(
            num_blocks=num_blocks,
            num_channels=num_channels,
            action_space_size=action_space_size,
            spatial_size=spatial_size,
            sequence_length=sequence_length,
        )

        # Decoder: latent -> reconstructed frame
        self.decoder = DecoderNetwork(
            observation_shape=observation_shape,
            num_blocks=num_blocks,
            num_channels=num_channels,
            upsample=True,
        )

        # Reward prediction head: latent -> reward
        self.reward_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels * spatial_size * spatial_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def encode(self, observations):
        """
        Encode observations to latent states.

        Args:
            observations: [batch, C, H, W] single frames
                      or [batch, timesteps, C, H, W] sequences

        Returns:
            latents: [batch, num_channels, H', W']
                  or [batch, timesteps, num_channels, H', W']
        """
        if observations.dim() == 5:
            # Sequence input: [batch, timesteps, C, H, W]
            batch, timesteps, C, H, W = observations.shape
            # Process each frame
            obs_flat = observations.view(batch * timesteps, C, H, W)
            latents_flat = self.encoder(obs_flat)
            # Reshape back
            _, c, h, w = latents_flat.shape
            return latents_flat.view(batch, timesteps, c, h, w)
        else:
            # Single frame: [batch, C, H, W]
            return self.encoder(observations)

    def predict_dynamics(self, latent_states, actions):
        """
        Predict next latent states given current states and actions.

        Args:
            latent_states: [batch, timesteps, num_channels, H', W']
            actions: [batch, timesteps] or [batch, timesteps, action_dim]

        Returns:
            predicted_latents: [batch, timesteps, num_channels, H', W']
        """
        return self.dynamics(latent_states, actions)

    def decode(self, latent_states):
        """
        Decode latent states to observations.

        Args:
            latent_states: [batch, num_channels, H', W'] single states
                        or [batch, timesteps, num_channels, H', W'] sequences

        Returns:
            observations: [batch, C, H, W] or [batch, timesteps, C, H, W]
        """
        if latent_states.dim() == 5:
            # Sequence: [batch, timesteps, C, H', W']
            batch, timesteps, c, h, w = latent_states.shape
            latents_flat = latent_states.view(batch * timesteps, c, h, w)
            obs_flat = self.decoder(latents_flat)
            _, C, H, W = obs_flat.shape
            return obs_flat.view(batch, timesteps, C, H, W)
        else:
            return self.decoder(latent_states)

    def predict_reward(self, latent_states):
        """
        Predict rewards from latent states.

        Args:
            latent_states: [batch, num_channels, H', W'] single states
                        or [batch, timesteps, num_channels, H', W'] sequences

        Returns:
            rewards: [batch, 1] or [batch, timesteps, 1]
        """
        if latent_states.dim() == 5:
            batch, timesteps, c, h, w = latent_states.shape
            latents_flat = latent_states.view(batch * timesteps, c, h, w)
            rewards_flat = self.reward_head(latents_flat)
            return rewards_flat.view(batch, timesteps, 1)
        else:
            return self.reward_head(latent_states)

    def forward(self, observations, actions):
        """
        Full forward pass for training.

        Args:
            observations: [batch, timesteps, C, H, W] observation sequences
            actions: [batch, timesteps] action sequences

        Returns:
            dict containing:
                - predicted_latents: [batch, timesteps, num_channels, H', W']
                - predicted_observations: [batch, timesteps, C, H, W]
                - predicted_rewards: [batch, timesteps, 1]
                - encoded_latents: [batch, timesteps, num_channels, H', W']
        """
        # Encode observations to latent
        encoded_latents = self.encode(observations)

        # Predict next latents with dynamics + STU
        predicted_latents = self.predict_dynamics(encoded_latents, actions)

        # Decode to observations
        predicted_observations = self.decode(predicted_latents)

        # Predict rewards
        predicted_rewards = self.predict_reward(predicted_latents)

        return {
            "predicted_latents": predicted_latents,
            "predicted_observations": predicted_observations,
            "predicted_rewards": predicted_rewards,
            "encoded_latents": encoded_latents,
        }
