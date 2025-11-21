from .agents import (
    RandomAgent,
    CorrelatedNoiseAgent,
    StructuredExplorationAgent,
    MixedExplorationAgent,
    ZeroAgent,
)
from .envs import Env, DummyEnv
from .utils.printing import Task, progress
from .utils.plotting import pred_vs_true_plot
from .models import Model
from .blocks import MLPBlock, SpectralBlock, MLPSettings, SpectralSettings

__all__ = [
    "Env",
    "DummyEnv",
    "RandomAgent",
    "CorrelatedNoiseAgent",
    "StructuredExplorationAgent",
    "MixedExplorationAgent",
    "Task",
    "progress",
    "pred_vs_true_plot",
    "Model",
    "MLPBlock",
    "SpectralBlock",
    "MLPSettings",
    "SpectralSettings",
    "ZeroAgent",    
]

__version__ = "0.1.0"