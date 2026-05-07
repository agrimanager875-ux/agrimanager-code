"""Base environment configuration class.

This module defines a minimal base configuration class that can be easily
extended by specific environments.
"""

from typing import Any, Dict, Optional


class BaseEnvConfig:
    """Minimal base configuration class for environments.

    This class can be extended by specific environment configurations
    to add their own parameters.

    Attributes:
        seed: Random seed for reproducibility
        llm_mode: Whether the environment should expose its natural-language interface
        turn_num: Number of decision steps per episode
        **kwargs: Additional environment-specific parameters
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        llm_mode: bool = True,
        turn_num: int = 1,
        **kwargs,
    ):
        """Initialize the base environment configuration.

        Args:
            seed: Random seed for reproducibility.
            llm_mode: ``True`` when the environment should expose a natural-language
                interface (observations become prompts, actions can be free-form text);
                ``False`` when it should operate in its native numeric form.
            turn_num: Number of steps per episode.
            **kwargs: Additional environment-specific parameters that will be stored
                as attributes on the config object.
        """
        self.seed = seed
        self.llm_mode = llm_mode
        self.turn_num = turn_num

        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return vars(self)
