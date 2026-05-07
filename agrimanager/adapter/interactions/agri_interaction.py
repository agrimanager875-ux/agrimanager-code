"""AgriInteraction for VERL ToolAgentLoop integration.

This module provides an Interaction wrapper that bridges AgriManager
environments with VERL's ToolAgentLoop training framework.
"""

from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction
from agrimanager.env.base import create_environment


class AgriInteraction(BaseInteraction):
    """Interaction wrapper for AgriManager environments.

    This class implements the BaseInteraction interface required by VERL's
    ToolAgentLoop. It wraps AgriManager environments (e.g., WOFOSTEnv) and
    handles the translation between LLM messages and environment actions.

    The interaction flow:
    1. start_interaction: Creates an environment instance and returns initial state
    2. generate_response: Parses LLM actions, executes env.step(), returns observations
    3. finalize_interaction: Cleans up environment resources

    Attributes:
        config: Configuration dict from interaction config
        _instance_dict: Maps instance_id to environment state
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the AgriInteraction.

        Args:
            config: Configuration dict containing:
                - name: Interaction name (default: "agri")
                - default_env_type: Default environment type (default: "wofost_gym")
        """
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.default_env_type = config.get("default_env_type", "wofost_gym")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        env_config: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Create an environment instance and get initial observation.

        Args:
            instance_id: Optional unique identifier for this interaction instance.
                        If None, a UUID will be generated.
            env_config: Environment configuration dict. Should contain all
                       parameters needed to create the environment (e.g., year,
                       env_id, agro_file, etc.). The env_name can be specified
                       in env_config or defaults to self.default_env_type.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            instance_id: The unique identifier for this interaction instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Copy before popping env_name so rollout metadata keeps the original
        # env type for later per-turn training/regeneration passes.
        env_config = deepcopy(env_config or {})
        env_name = env_config.pop("env_name", self.default_env_type)

        # Create environment using AgriManager's factory function
        env, _ = create_environment(env_name, env_config)
        obs, info = env.reset()

        self._instance_dict[instance_id] = {
            "env": env,
            "obs": obs,
            "done": False,
            "turn_metrics": info.get("turn_metrics", {}),
            "step_count": 0,
        }

        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Parse action from LLM response, execute environment step, return observation.

        This method:
        1. Extracts the last assistant message from the conversation
        2. Passes it to env.step() which parses <answer>...</answer> format
        3. Returns the new observation and reward

        Args:
            instance_id: The interaction instance identifier.
            messages: List of conversation messages. Each message is a dict with
                     'role' (system/user/assistant) and 'content' keys.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A tuple of (should_terminate, response, reward, additional_data):
            - should_terminate: True if the episode has ended
            - response: The new observation (natural language prompt)
            - reward: The reward for this step
            - additional_data: Dict containing metrics and any additional info
        """
        state = self._instance_dict[instance_id]
        env = state["env"]

        # If already done, return the final state
        if state["done"]:
            return True, state["obs"], 0.0, {
                "turn_metrics": state["turn_metrics"],
                "trajectory_metrics": self._get_env_trajectory_metrics(env),
            }

        # Extract the last assistant message content
        content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        # Execute environment step
        obs, reward, done, info = env.step(content)

        # Update instance state
        state["obs"] = obs
        state["done"] = done
        state["turn_metrics"] = info.get("turn_metrics", state["turn_metrics"])
        state["step_count"] += 1

        return done, obs, reward, {
            "turn_metrics": info.get("turn_metrics", {}),
            "trajectory_metrics": info.get("trajectory_metrics", {}),
        }

    def _get_env_trajectory_metrics(self, env) -> dict[str, Any]:
        getter = getattr(env, "get_trajectory_metrics", None)
        if callable(getter):
            metrics = getter()
            return metrics if isinstance(metrics, dict) else {}
        return {}

    async def get_trajectory_metrics(self, instance_id: str) -> dict[str, Any]:
        """Return current episode-level metrics, even for max-turn truncation."""
        state = self._instance_dict[instance_id]
        return self._get_env_trajectory_metrics(state["env"])

    async def get_current_observation(self, instance_id: str) -> str:
        """Return current observation without stepping the env."""
        return self._instance_dict[instance_id]["obs"]

    async def get_system_prompt(self, instance_id: str) -> str:
        """Return the system prompt generated by the environment."""
        return self._instance_dict[instance_id]["env"].system_prompt()

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this interaction instance.

        Args:
            instance_id: The interaction instance identifier.
            **kwargs: Additional keyword arguments (ignored).
        """
        if instance_id in self._instance_dict:
            state = self._instance_dict[instance_id]
            if "env" in state and state["env"] is not None:
                state["env"].close()
            del self._instance_dict[instance_id]
