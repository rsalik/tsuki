# Multi-Layer Perceptron (MLP) model building block
from collections.abc import Callable
from dataclasses import dataclass
from typing import Tuple
from .base import Block, BlockSettings
import numpy as np
from scipy.linalg import eigh as largest_eigh

import torch
import torch.nn as nn

from .decorators import historical


@dataclass
class SpectralSettings(BlockSettings):
    """Settings for the Spectral block."""

    out_dim: int = 32
    num_filters: int = 24
    spectral_history_length: int = 100


@dataclass
@historical
class SpectralBlock(Block):
    settings: SpectralSettings

    def __init__(self, settings: SpectralSettings):
        super().__init__(settings)

    def setup(self, internal_block, in_dim, out_dim=None):
        if out_dim is None:
            out_dim = self.settings.out_dim

        # in_dim at this point is already multiplied by history_length from HistoryConcat
        # So we need to recover the original feature dimension
        original_dim = in_dim // self.settings.spectral_history_length
        
        internal_block.filters = get_filters(
            self.settings.spectral_history_length, self.settings.num_filters
        )  # (spectral_history_length, num_filters)
        
        internal_block.original_dim = original_dim
        internal_block.history_length = self.settings.spectral_history_length

        # Learn a linear layer to map from spectral features to output dimension
        internal_block.out_dim = out_dim
        internal_block.linear = nn.Linear(self.settings.num_filters * original_dim, out_dim)

    def forward(self, internal_block, x):
        # x comes from HistoryConcat with shape (batch, history_length * original_dim) or (history_length * original_dim,)
        # We need to reshape it to (history_length, original_dim) or (batch, history_length, original_dim)
        
        if x.dim() == 1:
            # Shape: (history_length * original_dim,) -> (history_length, original_dim)
            x = x.reshape(internal_block.history_length, internal_block.original_dim)
            # Apply spectral filters: (num_filters, history_length) @ (history_length, original_dim) -> (num_filters, original_dim)
            spectral_features = torch.einsum("fh,hd->fd", internal_block.filters.T, x)
            # Flatten: (num_filters, original_dim) -> (num_filters * original_dim,)
            spectral_features = spectral_features.reshape(-1)
        else:
            # Shape: (batch, history_length * original_dim) -> (batch, history_length, original_dim)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, internal_block.history_length, internal_block.original_dim)
            # Apply spectral filters: (num_filters, history_length) @ (batch, history_length, original_dim) -> (batch, num_filters, original_dim)
            spectral_features = torch.einsum("fh,bhd->bfd", internal_block.filters.T, x)
            # Flatten: (batch, num_filters, original_dim) -> (batch, num_filters * original_dim)
            spectral_features = spectral_features.reshape(batch_size, -1)

        return internal_block.linear(spectral_features)

    @property
    def history_length(self) -> int:
        return self.settings.spectral_history_length


def get_filters(spectral_history_length: int, num_filters: int = 24) -> torch.Tensor:
    num_filters = min(num_filters, spectral_history_length)
    i = torch.arange(1, spectral_history_length + 1, dtype=torch.float32)
    i_plus_j = i[None, :] + i[:, None]
    Z = 2 / (i_plus_j**3 - i_plus_j)
    evals, evecs = torch.linalg.eigh(Z)
    return torch.flip(evecs, [0])[:, -num_filters:] * (evals[-num_filters:] ** 0.25)