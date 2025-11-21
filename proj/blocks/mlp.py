# Multi-Layer Perceptron (MLP) model building block
from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor
from .base import Block, BlockSettings

import torch.nn as nn

@dataclass
class MLPSettings(BlockSettings):
    """Settings for the MLP block."""

    num_layers: int = 2
    hidden_dim: int = 32
    out_dim: int = 32
    activation: Callable[[float], float] = nn.ReLU()

@dataclass
class MLPBlock(Block):
    """Multi-Layer Perceptron (MLP) block using PyTorch"""
    settings: MLPSettings

    def __init__(self, settings: MLPSettings):
        super().__init__(settings)

    def setup(self, internal_block, in_dim, out_dim=None):
        # Out_dim overriden if block comes last in model
        if out_dim is None:
            out_dim = self.settings.out_dim

        layers = []
        current_dim = in_dim

        if self.settings.num_layers < 2:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self.settings.activation)
        else:
            for _ in range(self.settings.num_layers - 1):
                layers.append(nn.Linear(current_dim, self.settings.hidden_dim))
                layers.append(self.settings.activation)
                current_dim = self.settings.hidden_dim

            layers.append(nn.Linear(current_dim, out_dim))

        internal_block.model = nn.Sequential(*layers)
        internal_block.out_dim = out_dim

    def forward(self, internal_block, x: Tensor):
        return internal_block.model(x)

    @property
    def out_dim(self) -> int:
        return self.settings.hidden_dim
