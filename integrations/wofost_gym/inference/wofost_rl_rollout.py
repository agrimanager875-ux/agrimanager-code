"""PPO/DQN/SAC agent rollout script for WOFOST-Gym evaluation."""

import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
import hydra
import json
import random
import numpy as np
import torch
from omegaconf import OmegaConf

# /.../AgriManager/integrations/wofost_gym/inference/wofost_rl_rollout.py
# -> repo root is parents[4] (/.../AgriManager)
agrimanager_root = Path(__file__).resolve().parents[4]
if str(agrimanager_root) not in sys.path:
    sys.path.insert(0, str(agrimanager_root))

from agrimanager.env.base import create_environment, load_env_configs_from_parquet
from agrimanager.env.wofost_gym.crop_traits_observation import (
    CropTraitEncoder,
    append_crop_traits,
    extend_observation_space,
)


def load_dataset(test_files: str) -> tuple:
    """Load dataset configurations from parquet file.

    Returns:
        (configs, env_name) — env_name is parsed from the path or data_source.
    """
    configs, env_name, dataset_path = load_env_configs_from_parquet(test_files)
    print(f"Loading dataset from: {dataset_path}")
    print(f"Loaded {len(configs)} environment configurations")
    return configs, env_name


def create_env(env_name: str, env_config: Dict[str, Any]):
    """Create environment instance with llm_mode=False for gym interface."""
    env_config = env_config.copy()
    env_config['llm_mode'] = False
    return create_environment(env_name, env_config)


def infer_runtime_features(env_config: Dict[str, Any]) -> dict[str, Any]:
    return {
        "include_crop_traits": bool(env_config.get("include_crop_traits", False)),
        "crop_traits_dir": env_config.get("crop_traits_dir"),
        "trait_schema": env_config.get("trait_schema"),
        "wofost_gym_path": env_config.get("wofost_gym_path"),
    }


def get_action_space(env):
    """Get the action space from environment."""
    if hasattr(env, 'env') and hasattr(env.env, 'action_space'):
        return env.env.action_space
    elif hasattr(env, 'action_space'):
        return env.action_space
    else:
        raise AttributeError("Environment does not have an action_space attribute")


def get_observation_space(env):
    """Get the observation space from the inner gym environment."""
    if hasattr(env, 'env') and hasattr(env.env, 'observation_space'):
        return env.env.observation_space
    elif hasattr(env, 'observation_space'):
        return env.observation_space
    else:
        raise AttributeError("Environment does not have an observation_space attribute")


class MockVecEnv:
    """Mock vectorized environment providing single_observation_space/single_action_space.

    PPO/DQN/SAC agents from WOFOSTGym expect a vectorized env interface for
    __init__. This mock provides just enough to instantiate the agent.
    """

    def __init__(self, observation_space, action_space):
        self.single_observation_space = observation_space
        self.single_action_space = action_space


def load_agent(
    agent_path: str,
    agent_type: str,
    observation_space,
    action_space,
    device: str,
    wofost_gym_path: str,
):
    """Load a trained RL agent from agent.pt.

    Args:
        agent_path: Path to the saved agent state dict (agent.pt)
        agent_type: Agent class name (PPO, DQN, SAC)
        observation_space: Gym observation space
        action_space: Gym action space
        device: Torch device string

    Returns:
        Loaded agent in eval mode
    """
    # Add WOFOSTGym to path for importing agent classes
    if wofost_gym_path not in sys.path:
        sys.path.insert(0, wofost_gym_path)

    # Import the agent class dynamically
    module = importlib.import_module(f"rl_algs.{agent_type}")
    agent_cls = getattr(module, agent_type)

    # Create mock vectorized env for agent initialization
    mock_envs = MockVecEnv(observation_space, action_space)

    # Instantiate and load weights
    agent = agent_cls(mock_envs)
    state_dict = torch.load(agent_path, map_location=device, weights_only=True)
    agent.load_state_dict(state_dict)
    agent.to(device)
    agent.eval()

    return agent


def create_normalizer(env):
    """Create a NormalizeObservation wrapper for normalization only.

    We wrap the inner gym env to get access to the normalize() method,
    but we don't use it for stepping — WOFOSTEnv.step() handles turn_metrics
    extraction from raw observations before normalization.
    """
    from pcse_gym.wrappers import NormalizeObservation
    return NormalizeObservation(env.env)


def extract_crop_name(env_config: Dict[str, Any]) -> str:
    """Extract crop name from env_config.agro_file."""
    agro_file = Path(str(env_config.get("agro_file", "") or "")).name
    if agro_file.endswith("_agro.yaml"):
        return agro_file[:-10]
    if agro_file.endswith(".yaml"):
        return agro_file[:-5]
    return agro_file or "unknown"


def sanitize_filename(name: str) -> str:
    """Turn a crop name into a stable filename component."""
    cleaned = [
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in (name or "unknown").strip().lower()
    ]
    sanitized = "".join(cleaned).strip("._")
    return sanitized or "unknown"


def save_results(output_dir: str, results: List[Dict[str, Any]], split_by_crop: bool = False):
    """Save overall rollout results and optionally per-crop result files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")

    if not split_by_crop:
        return

    by_crop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        crop = extract_crop_name(item.get("env_config") or {})
        by_crop[crop].append(item)

    crop_output_dir = output_path / "results_by_crop"
    crop_output_dir.mkdir(parents=True, exist_ok=True)

    crop_index = []
    for crop, crop_results in sorted(by_crop.items()):
        crop_file = crop_output_dir / f"{sanitize_filename(crop)}.json"
        with open(crop_file, "w") as f:
            json.dump(crop_results, f, indent=2)

        crop_index.append({
            "crop": crop,
            "num_envs": len(crop_results),
            "results_file": str(crop_file.relative_to(output_path)),
        })
        print(f"Per-crop results saved to {crop_file} ({len(crop_results)} envs)")

    index_file = crop_output_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(crop_index, f, indent=2)

    print(f"Per-crop index saved to {index_file}")


def run_rl_rollout(
    output_dir: str,
    agent_path: str,
    agent_type: str = "PPO",
    seed: int = 0,
    device: str = "cpu",
    split_by_crop: bool = False,
    include_crop_traits: bool = False,
    crop_traits_dir: str | None = None,
    crop_trait_schema: str | None = None,
    *,
    test_files: str | None = None,
    env_configs: List[Dict[str, Any]] | None = None,
    env_name: str = "wofost_gym",
):
    """Run trained RL agent rollout on parquet test set.

    Args:
        output_dir: Directory to save results
        agent_path: Path to agent.pt checkpoint
        agent_type: Agent class name (PPO, DQN, SAC)
        seed: Random seed
        device: Torch device
        split_by_crop: Save additional per-crop result files
        include_crop_traits: Append numeric crop-trait features to NN observations
        crop_traits_dir: Optional directory containing crop-trait JSON cards
        crop_trait_schema: Schema name used to load numeric crop-trait cards
    """
    print("=" * 80)
    print(f"{agent_type} Agent Rollout Configuration")
    print("=" * 80)
    if test_files:
        print(f"Test files: {test_files}")
    else:
        print("Dataset source: in-memory dataset artifact")
    print(f"Output directory: {output_dir}")
    print(f"Agent path: {agent_path}")
    print(f"Agent type: {agent_type}")
    print(f"Random seed: {seed}")
    print(f"Device: {device}")
    print(f"Split results by crop: {split_by_crop}")
    print(f"Include crop traits: {include_crop_traits}")
    print(f"Crop trait schema: {crop_trait_schema or 'default'}")
    print("=" * 80)

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    if env_configs is None:
        if not test_files:
            raise ValueError("run_rl_rollout requires test_files or env_configs.")
        env_configs, env_name = load_dataset(test_files)
    num_envs = len(env_configs)
    trait_encoder = (
        CropTraitEncoder(
            traits_dir=crop_traits_dir,
            trait_schema=crop_trait_schema,
        )
        if include_crop_traits
        else None
    )
    if trait_encoder is not None:
        print(
            "Crop-trait observation features: "
            f"schema={trait_encoder.trait_schema}, dim={trait_encoder.dim}, "
            f"traits_dir={trait_encoder.traits_dir}, "
            f"crops={list(trait_encoder.crop_names)}"
        )

    # Create environments
    print(f"\n[2/5] Creating {num_envs} environments...")
    envs = []
    configs = []
    action_spaces = []
    normalizers = []

    for i, env_config in enumerate(tqdm(env_configs, desc="Creating envs")):
        env, config = create_env(env_name, env_config)
        envs.append(env)
        configs.append(config)
        action_spaces.append(get_action_space(env))
        normalizers.append(create_normalizer(env))

    # Get turn_num from first config
    turn_num = configs[0].turn_num
    obs_space = get_observation_space(envs[0])
    if trait_encoder is not None:
        obs_space = extend_observation_space(obs_space, trait_encoder)
    act_space = action_spaces[0]
    print(f"Episodes will run for {turn_num} turns")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")

    # Load agent
    print(f"\n[3/5] Loading {agent_type} agent from {agent_path}...")
    wofost_gym_path = str(getattr(configs[0], "wofost_gym_path", "")) if configs else ""
    if not wofost_gym_path:
        raise ValueError("Resolved WOFOST config is missing wofost_gym_path.")
    agent = load_agent(agent_path, agent_type, obs_space, act_space, device, wofost_gym_path)
    print(f"Agent loaded successfully")

    # Initialize storage for results
    results = []
    for i in range(num_envs):
        results.append({
            'env_id': i,
            'env_config': env_configs[i],
            'turns': []
        })

    # Reset all environments
    print(f"\n[4/5] Resetting all environments...")
    raw_observations = []
    for i, env in enumerate(tqdm(envs, desc="Resetting")):
        raw_obs, info = env.reset()
        metrics = (info or {}).get('turn_metrics', {})
        raw_observations.append(raw_obs)

        # Record initial state
        results[i]['turns'].append({
            'turn': 0,
            'action_id': None,
            'turn_metrics': metrics,
        })

    # Run rollout
    print(f"\n[5/5] Running {agent_type} rollout for {turn_num} turns...")
    done_flags = [False] * num_envs
    for turn in tqdm(range(1, turn_num + 1), desc="Turns"):
        active_indices = [i for i, done in enumerate(done_flags) if not done]
        if not active_indices:
            print(f"\nAll environments finished at turn {turn - 1}")
            break

        for i in active_indices:
            try:
                # Normalize observation and get agent action
                normalized_obs = normalizers[i].normalize(
                    np.array([raw_observations[i]])
                )[0]
                if trait_encoder is not None:
                    normalized_obs = append_crop_traits(
                        normalized_obs,
                        extract_crop_name(env_configs[i]),
                        trait_encoder,
                    )
                obs_tensor = torch.from_numpy(normalized_obs).float().to(device)

                with torch.no_grad():
                    action = agent.get_action(obs_tensor)
                action_id = action.item()

                # Step environment with action
                raw_obs, _, done, info = envs[i].step(action_id)
                metrics = (info or {}).get('turn_metrics', {})
                raw_observations[i] = raw_obs

                # Store results
                results[i]['turns'].append({
                    'turn': turn,
                    'action_id': int(action_id),
                    'turn_metrics': metrics,
                })

                if done:
                    done_flags[i] = True

            except Exception as e:
                print(f"\nError in env {i} at turn {turn}: {e}")
                results[i]['turns'].append({
                    'turn': turn,
                    'action_id': None,
                    'error': str(e),
                    'turn_metrics': {}
                })

        if all(done_flags):
            print(f"\nAll environments finished at turn {turn}")
            break

    # Save results
    print(f"\nSaving results to {output_dir}...")
    save_results(output_dir=output_dir, results=results, split_by_crop=split_by_crop)
    print(f"\n{agent_type} rollout completed!")


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(config) -> None:
    OmegaConf.resolve(config)
    cfg = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config, got {type(cfg)!r}")

    data_cfg = cfg.get("data") or {}
    runtime_cfg = cfg.get("runtime") or {}
    output_cfg = cfg.get("output") or {}
    agent_cfg = cfg.get("agent") or {}
    has_include_crop_traits_override = (
        "include_crop_traits" in runtime_cfg
        or "include_crop_traits" in agent_cfg
        or "include_crop_traits" in cfg
    )
    has_crop_traits_dir_override = (
        "crop_traits_dir" in runtime_cfg
        or "crop_traits_dir" in agent_cfg
        or "crop_traits_dir" in cfg
    )
    has_crop_trait_schema_override = (
        "crop_trait_schema" in runtime_cfg
        or "crop_trait_schema" in agent_cfg
        or "crop_trait_schema" in cfg
        or "trait_schema" in cfg
    )

    test_files = data_cfg.get("inference_file") or cfg.get("test_files")
    output_dir = output_cfg.get("dir") or cfg.get("output_dir")
    agent_path = agent_cfg.get("path") or cfg.get("agent_path")
    agent_type = agent_cfg.get("type") or cfg.get("agent_type", "PPO")
    seed = int(runtime_cfg.get("seed", cfg.get("seed", 0)))
    device = str(runtime_cfg.get("device", cfg.get("device", "cpu")))
    split_by_crop = bool(runtime_cfg.get("split_by_crop", cfg.get("split_by_crop", False)))
    include_crop_traits = bool(
        runtime_cfg.get("include_crop_traits", agent_cfg.get("include_crop_traits", cfg.get("include_crop_traits", False)))
    )
    crop_traits_dir = (
        runtime_cfg.get("crop_traits_dir")
        or agent_cfg.get("crop_traits_dir")
        or cfg.get("crop_traits_dir")
    )
    crop_trait_schema = (
        runtime_cfg.get("crop_trait_schema")
        or agent_cfg.get("crop_trait_schema")
        or cfg.get("crop_trait_schema")
        or cfg.get("trait_schema")
    )

    if not agent_path:
        raise ValueError("Config must specify agent.path.")
    if not test_files:
        raise ValueError("Config must specify data.inference_file.")
    if not output_dir:
        raise ValueError("Config must specify output.dir.")

    env_configs, env_name = load_dataset(str(test_files))
    runtime_features = infer_runtime_features(env_configs[0]) if env_configs else {}
    if not has_include_crop_traits_override:
        include_crop_traits = bool(runtime_features["include_crop_traits"])
    if not has_crop_traits_dir_override and crop_traits_dir is None:
        crop_traits_dir = runtime_features["crop_traits_dir"]
    if not has_crop_trait_schema_override and crop_trait_schema is None:
        crop_trait_schema = runtime_features["trait_schema"]

    run_rl_rollout(
        test_files=str(test_files),
        output_dir=str(output_dir),
        agent_path=str(agent_path),
        agent_type=str(agent_type),
        seed=seed,
        device=device,
        split_by_crop=split_by_crop,
        include_crop_traits=include_crop_traits,
        crop_traits_dir=None if crop_traits_dir is None else str(crop_traits_dir),
        crop_trait_schema=None if crop_trait_schema is None else str(crop_trait_schema),
        env_configs=env_configs,
        env_name=env_name,
    )


if __name__ == "__main__":
    main()
