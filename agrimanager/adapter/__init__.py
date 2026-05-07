"""Adapter module for AgriManager.

This module provides the adapter layer between AgriManager environments and
VERL's ToolAgentLoop for reinforcement learning training.
"""

from .interactions.agri_interaction import AgriInteraction
from .reward.agri_reward import compute_score
from .utils import AgriGenerationsLogger

__all__ = [
    "AgriInteraction",
    "compute_score",
    "AgriTrainer",
    "AgriGenerationsLogger",
]


def __getattr__(name):
    if name == "AgriTrainer":
        from .trainer.trainer import AgriTrainer

        return AgriTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
