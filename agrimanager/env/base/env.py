"""Base environment class.

This module defines the base environment class that all environments
should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .env_config import BaseEnvConfig


class BaseEnv(ABC):
    """Base class for all environments.

    This abstract class defines the standard interface that all environments
    must implement. It follows a minimal Gym-like interface with additional
    support for LLM agents.

    Attributes:
        config: Environment configuration object
    """

    def __init__(self, config: BaseEnvConfig):
        """Initialize the base environment.

        Args:
            config: Environment configuration object
        """
        self.config = config
        self._setup()

    def _setup(self) -> None:
        """Setup the environment after initialization.

        This method can be overridden by subclasses to perform additional
        setup operations.
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to initial state.

        Returns:
            obs: Initial observation
            info: Dictionary containing additional information.
                ``turn_metrics`` may be provided for logging or diagnostics,
                but reward semantics should always flow through the scalar
                ``reward`` returned by :meth:`step`.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            obs: Observation after executing the action
            reward: Reward obtained from the action
            done: Whether the episode has ended
            info: Dictionary containing additional information.
                The reward contract is generic and step-based: environments
                should emit a scalar reward on every turn. If an environment
                conceptually uses a trajectory reward, it should represent it
                through this same stream, typically by returning zero rewards
                on earlier turns and placing the trajectory reward on the final
                turn.

                Optional logging/diagnostic fields may include:
                - ``turn_metrics``: per-step metrics
                - ``trajectory_metrics``: episode-level summary metrics
                  (often only when ``done=True``)
                - ``raw_llm_response`` when the action originated from a
                  model's free-form text
        """
        raise NotImplementedError

    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for LLM agents.

        Returns:
            System prompt string describing the environment, task, and
            expected behavior for the LLM agent
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment and cleanup resources.

        This method should be overridden by subclasses that need to
        perform cleanup operations.
        """
        pass
