from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BlockSettings:
    pass


@dataclass
class Block:

    def __init__(self, settings: BlockSettings):
        self.settings = settings

    @abstractmethod
    def forward(self, internal_block, x) -> Any:
        pass

    @abstractmethod
    def setup(self, internal_block, in_dim, out_dim=None):
        """Not meant to be called by the user. Internal method used by a model
        to initialize internal blocks."""
        pass

    def create(self, in_dim):
        """Not meant to be called by the user. Internal method used by a model
        to string internal blocks together."""

        internal_block = IntBlock()
        internal_block.in_dim = in_dim
        internal_block.forward = lambda x: self.forward(internal_block, x)
        return internal_block


class IntBlock:
    """An internal instance of a block actually within a model"""

    in_dim: int
    _out_dim: int = 0

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @out_dim.setter
    def out_dim(self, value: int):
        self._out_dim = value

    def forward(self, x):
        raise NotImplementedError
