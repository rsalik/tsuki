"""
D4RL-Atari Dataset Loader for World Model Training.

Provides sequence-based data loading for training world models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import d4rl_atari
    import gym

    D4RL_AVAILABLE = True
except ImportError:
    D4RL_AVAILABLE = False
    print("Warning: d4rl_atari not installed. Install with: pip install d4rl-atari")


class AtariSequenceDataset(Dataset):
    """
    Dataset that provides sequences of (observations, actions, rewards)
    for world model training.
    """

    def __init__(
        self,
        game: str = "breakout",
        dataset_type: str = "mixed",
        sequence_length: int = 16,
        frame_stack: int = 4,
    ):
        """
        Args:
            game: Atari game name (e.g., 'breakout', 'pong', 'seaquest')
            dataset_type: 'mixed', 'medium', or 'expert'
            sequence_length: Number of timesteps per training sequence
            frame_stack: Number of frames to stack as observation
        """
        self.sequence_length = sequence_length
        self.frame_stack = frame_stack

        if not D4RL_AVAILABLE:
            raise ImportError(
                "d4rl_atari is required. Install with: pip install d4rl-atari"
            )

        # Load D4RL-Atari dataset
        env_name = f"{game}-{dataset_type}-v0"
        env = gym.make(env_name)
        self.dataset = env.get_dataset()

        # Extract data
        self.observations = self.dataset[
            "observations"
        ]  # [N, 84, 84] or [N, C, 84, 84]
        self.actions = self.dataset["actions"]  # [N]
        self.rewards = self.dataset["rewards"]  # [N]
        self.terminals = self.dataset["terminals"]  # [N]

        # Handle observation shape
        if len(self.observations.shape) == 3:
            # [N, H, W] -> add channel dim
            self.observations = self.observations[:, np.newaxis, :, :]

        # Find valid sequence start indices (avoid crossing episode boundaries)
        self.valid_indices = self._compute_valid_indices()

        print(
            f"Loaded {env_name}: {len(self.observations)} frames, {len(self.valid_indices)} valid sequences"
        )

    def _compute_valid_indices(self):
        """Find indices where we can extract full sequences without crossing episodes."""
        valid = []
        total_len = len(self.observations)

        for i in range(total_len - self.sequence_length - self.frame_stack):
            # Check if any terminal in the sequence
            seq_end = i + self.sequence_length + self.frame_stack
            if not np.any(self.terminals[i:seq_end]):
                valid.append(i)

        return valid

    def __len__(self):
        return len(self.valid_indices)

    def _stack_frames(self, idx):
        """Stack frames for a single timestep."""
        frames = self.observations[
            idx : idx + self.frame_stack
        ]  # [frame_stack, C, H, W]
        return frames.reshape(
            -1, frames.shape[-2], frames.shape[-1]
        )  # [frame_stack * C, H, W]

    def __getitem__(self, idx):
        """
        Returns:
            observations: [sequence_length, frame_stack * C, H, W] - stacked frames
            actions: [sequence_length] - actions taken
            rewards: [sequence_length] - rewards received
            next_observations: [sequence_length, frame_stack * C, H, W] - next stacked frames
        """
        start_idx = self.valid_indices[idx]

        # Build sequences
        observations = []
        next_observations = []
        actions = []
        rewards = []

        for t in range(self.sequence_length):
            frame_idx = start_idx + t

            # Stack frames for current and next observation
            obs = self._stack_frames(frame_idx)
            next_obs = self._stack_frames(frame_idx + 1)

            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(self.actions[frame_idx + self.frame_stack - 1])
            rewards.append(self.rewards[frame_idx + self.frame_stack - 1])

        return {
            "observations": torch.FloatTensor(np.array(observations))
            / 255.0,  # Normalize to [0, 1]
            "actions": torch.LongTensor(np.array(actions)),
            "rewards": torch.FloatTensor(np.array(rewards)),
            "next_observations": torch.FloatTensor(np.array(next_observations)) / 255.0,
        }


def create_dataloader(
    game: str = "breakout",
    dataset_type: str = "mixed",
    sequence_length: int = 16,
    frame_stack: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """Create a DataLoader for world model training."""
    dataset = AtariSequenceDataset(
        game=game,
        dataset_type=dataset_type,
        sequence_length=sequence_length,
        frame_stack=frame_stack,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
