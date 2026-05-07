"""
dssat_gym environment wrapper.

This module provides a wrapper around DSSAT (via gym_dssat_pdi) to make it
compatible with the BaseEnv interface. It acts as a bridge between DSSAT
simulation and LLM agents by converting observations to natural language
prompts and parsing LLM responses back to actions.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from agrimanager.env.base import BaseEnv
from agrimanager.env.base.objective_prompt import profit_cost_params, profit_reward_scale
from .env_config import DSSATEnvConfig
from .prompt import DSSATPromptGenerator


def _ensure_legacy_gym_imports() -> None:
    """Expose Gymnasium under legacy `gym.*` module names when available."""

    try:
        gymnasium = importlib.import_module("gymnasium")
    except ModuleNotFoundError:
        try:
            import gym  # noqa: F401
            return
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "gym_dssat requires a Gym API package, but neither `gym` nor "
                "`gymnasium` is installed in the active AgriManager environment. "
                "For the smoke tests, install `gym==0.26.2 gymnasium stable-baselines3` "
                "into the active Conda env."
            ) from exc

    # `gym_dssat_pdi` still imports legacy `gym.*` modules. Prefer Gymnasium as
    # the backing implementation even if the deprecated `gym` package is
    # installed, which avoids the noisy NumPy 2 compatibility warning and keeps
    # the import surface aligned with the maintained API.
    sys.modules["gym"] = gymnasium
    for legacy_name, modern_name in (
        ("gym.spaces", "gymnasium.spaces"),
        ("gym.envs", "gymnasium.envs"),
        ("gym.envs.registration", "gymnasium.envs.registration"),
        ("gym.utils", "gymnasium.utils"),
        ("gym.utils.seeding", "gymnasium.utils.seeding"),
    ):
        sys.modules[legacy_name] = importlib.import_module(modern_name)


def _patch_dssat_local_zmq_bind(DssatPdi) -> None:
    """Bind DSSAT's control socket to loopback instead of all interfaces."""

    if getattr(DssatPdi, "_agrimanager_local_zmq_bind", False):
        return

    import zmq

    def _launch_server_local(self):
        self._zmq_context = zmq.Context()
        self._server = self._zmq_context.socket(zmq.PAIR)
        bind_host = os.environ.get("DSSAT_ZMQ_HOST", "127.0.0.1")
        bind_addr = f"tcp://{bind_host}"
        try:
            self._port = self._server.bind_to_random_port(bind_addr, max_tries=10000)
        except zmq.error.ZMQError as exc:
            raise RuntimeError(
                f"Failed to bind DSSAT control socket on {bind_addr}: {exc}"
            ) from exc

    DssatPdi._launch_server = _launch_server_local
    DssatPdi._agrimanager_local_zmq_bind = True


class _RandomGeneratorCompat:
    """Expose old RandomState methods expected by gym_dssat_pdi."""

    def __init__(self, rng):
        self._rng = rng

    def randint(self, *args, **kwargs):
        return self._rng.integers(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _patch_dssat_numpy_rng(DssatPdi) -> None:
    """Make old gym_dssat_pdi RNG calls work with modern Gym/NumPy."""

    if getattr(DssatPdi, "_agrimanager_numpy_rng_compat", False):
        return

    original_seed = DssatPdi.seed

    def _seed_with_randint(self, seed=None):
        result = original_seed(self, seed)
        rng = getattr(self, "_random_generator", None)
        if rng is not None and not hasattr(rng, "randint") and hasattr(rng, "integers"):
            self._random_generator = _RandomGeneratorCompat(rng)
        return result

    DssatPdi.seed = _seed_with_randint
    DssatPdi._agrimanager_numpy_rng_compat = True


def _ensure_dssat_pdi_bridge_importable() -> None:
    """Make the packed gym-dssat-pdi bridge importable inside Ray workers."""

    try:
        importlib.import_module("gym_dssat_pdi")
        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        Path(os.environ.get("DSSAT_PDI_BRIDGE_DIR", "")),
        repo_root.parent / "AgriManagerExternal" / "gym_dssat_pdi" / "gym-dssat-pdi",
        repo_root / "AgriManagerExternal" / "gym_dssat_pdi" / "gym-dssat-pdi",
        repo_root.parent.parent / "AgriManagerExternal" / "gym_dssat_pdi" / "gym-dssat-pdi",
    ]

    seen_paths: set[str] = set()
    for path in candidate_paths:
        if not path:
            continue
        bridge_dir = path / "gym_dssat_pdi"
        path_str = str(path)
        if bridge_dir.is_dir() and path_str not in seen_paths:
            seen_paths.add(path_str)
            sys.path.insert(0, path_str)

    importlib.import_module("gym_dssat_pdi")


def _get_dssat_spack_search_roots(spack_root: str) -> list[Path]:
    """Return raw and resolved Spack runtime roots for relocated deployments."""

    root = Path(spack_root)
    resolved_root = root.resolve()
    candidates = [
        root,
        root.parent,
        root.parent / "opt" / "spack",
        resolved_root,
        resolved_root.parent,
        resolved_root.parent / "opt" / "spack",
    ]
    search_roots: list[Path] = []
    seen_roots: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen_roots:
            seen_roots.add(candidate_str)
            search_roots.append(candidate)
    return search_roots


def _get_dssat_subprocess_pythonpath(spack_root: str) -> str:
    """Build a Python site-packages path for the DSSAT subprocess only."""

    override = os.environ.get("DSSAT_SPACK_PYTHONPATH", "").strip()
    if override:
        return override

    search_roots = _get_dssat_spack_search_roots(spack_root)

    candidates: list[Path] = []
    seen_candidates: set[str] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("**/lib/python3.10/site-packages")):
            if not path.is_dir():
                continue
            path_str = str(path)
            if path_str not in seen_candidates:
                seen_candidates.add(path_str)
                candidates.append(path)

    # Put the packages that actually contain DSSAT's Python bridge first so the
    # subprocess can import `pdi` before falling back to other Spack deps like
    # pyzmq, packaging, etc.
    def _priority(path: Path) -> tuple[int, str]:
        has_bridge = (path / "pdi").exists() or (path / "gym_dssat_pdi").exists()
        return (0 if has_bridge else 1, str(path))

    paths = [str(path) for path in sorted(candidates, key=_priority)]
    return ":".join(paths)


def _get_dssat_subprocess_pythonhome(spack_root: str) -> str:
    """Find the relocated Spack Python prefix for DSSAT's embedded Python."""

    override = os.environ.get("DSSAT_SPACK_PYTHONHOME", "").strip()
    if override:
        return override

    search_roots = _get_dssat_spack_search_roots(spack_root)

    candidates: list[Path] = []
    seen_candidates: set[str] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for encodings in sorted(base.glob("**/lib/python3.10/encodings/__init__.py")):
            if not encodings.is_file():
                continue
            prefix = encodings.parents[3]
            prefix_str = str(prefix)
            if prefix.is_dir() and prefix_str not in seen_candidates:
                seen_candidates.add(prefix_str)
                candidates.append(prefix)

    def _priority(prefix: Path) -> tuple[int, str]:
        return (0 if prefix.name.startswith("python-") else 1, str(prefix))

    return str(sorted(candidates, key=_priority)[0]) if candidates else ""

class DSSATEnv(BaseEnv):
    """Wrapper for gym-dssat environment with LLM interface."""

    def __init__(self, config: DSSATEnvConfig):

        self.config = config

        # Allow external gym-dssat-pdi path
        if config.dssat_gym_path and config.dssat_gym_path not in sys.path:
            sys.path.insert(0, config.dssat_gym_path)

        _ensure_dssat_pdi_bridge_importable()
        _ensure_legacy_gym_imports()

        # Import actual DSSAT worker
        from gym_dssat_pdi.envs.dssat_pdi import DssatPdi
        _patch_dssat_local_zmq_bind(DssatPdi)
        _patch_dssat_numpy_rng(DssatPdi)

        env_params = config.env_params or {}

        # Resolve run_dssat from DSSAT_GYM_PATH env var or config
        dssat_gym_path = os.environ.get("DSSAT_GYM_PATH") or config.dssat_gym_path or ""
        run_dssat_path = os.path.join(dssat_gym_path, "bin", "run_dssat") if dssat_gym_path else ""

        # Ensure run_dssat exists
        if not run_dssat_path or not os.path.isfile(run_dssat_path):
            raise RuntimeError(
                f"run_dssat not found at: {run_dssat_path!r}\n"
                "Set the DSSAT_GYM_PATH environment variable to your spack gym-dssat-pdi installation."
            )

        # Remove unsupported flags
        if "PDI_EARLY_STOPPING" in env_params:
            print("⚠️ Removing unsupported PDI_EARLY_STOPPING param")
            env_params.pop("PDI_EARLY_STOPPING")

        # ------------------------------------------------------------
        # Force PDI environment variables so the DSSAT subprocess can
        # find its shared libraries and plugins regardless of how the
        # Ray worker was spawned.
        # ------------------------------------------------------------
        dssat_bin = os.path.dirname(run_dssat_path)
        spack_root = os.path.dirname(dssat_bin)  # …/spack/gym-dssat-pdi
        pdi_lib = os.path.join(spack_root, "lib")
        pdi_lib64 = os.path.join(spack_root, "lib64")
        conda_lib = os.path.join(sys.prefix, "lib")
        dssat_pythonhome = _get_dssat_subprocess_pythonhome(spack_root)
        dssat_pythonhome_lib = os.path.join(dssat_pythonhome, "lib") if dssat_pythonhome else ""

        os.environ.setdefault("PDI_PLUGIN_PATH", pdi_lib)
        for d in [conda_lib, dssat_pythonhome_lib, pdi_lib64, pdi_lib]:
            if os.path.isdir(d):
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                if d not in ld.split(":"):
                    os.environ["LD_LIBRARY_PATH"] = f"{d}:{ld}" if ld else d

        # Do not add Spack's Python 3.10 site-packages to PYTHONPATH here.
        # AgriManager runs under Conda Python 3.12, and leaking Spack's
        # Python-specific site-packages into the main process causes ABI
        # mismatches (for example importing Python 3.10-built NumPy from a
        # Python 3.12 worker). The DSSAT subprocess is launched via
        # `run_dssat` and only needs the shared-library/plugin paths above.

        # ------------------------------------------------------------
        # Enable DSSAT stdout/stderr logging so crashes are visible
        # instead of silently swallowed by /dev/null.
        # ------------------------------------------------------------
        if "log_saving_path" not in env_params:
            import tempfile
            log_dir = os.path.join(spack_root, "dssat_logs")
            os.makedirs(log_dir, exist_ok=True)
            env_params["log_saving_path"] = os.path.join(log_dir, f"dssat_{os.getpid()}.log")
            print(f"DSSAT log: {env_params['log_saving_path']}")

        # ------------------------------------------------------------
        # Create actual DSSAT simulator
        # ------------------------------------------------------------
        # DSSATPRO.L48 must be in DSSAT's working directory so it can
        # resolve absolute paths to crop/weather data files.
        profile_file = os.path.join(dssat_bin, "DSSATPRO.L48")
        if "auxiliary_file_paths" not in env_params and os.path.isfile(profile_file):
            env_params["auxiliary_file_paths"] = [profile_file]
        elif "auxiliary_file_paths" in env_params and os.path.isfile(profile_file):
            if profile_file not in env_params["auxiliary_file_paths"]:
                env_params["auxiliary_file_paths"].append(profile_file)

        num_seasons = getattr(config, "num_seasons", 1)
        orig_pythonpath = os.environ.get("PYTHONPATH")
        orig_pythonhome = os.environ.get("PYTHONHOME")
        dssat_pythonpath = _get_dssat_subprocess_pythonpath(spack_root)
        if dssat_pythonpath:
            os.environ["PYTHONPATH"] = (
                f"{dssat_pythonpath}:{orig_pythonpath}" if orig_pythonpath else dssat_pythonpath
            )
        if dssat_pythonhome:
            os.environ["PYTHONHOME"] = dssat_pythonhome
        try:
            dssat_kwargs = {
                "run_dssat_location": run_dssat_path,
                **env_params,
            }
            if "num_seasons" in inspect.signature(DssatPdi.__init__).parameters:
                dssat_kwargs["num_seasons"] = num_seasons
            elif num_seasons != 1:
                raise RuntimeError(
                    "The installed gym_dssat_pdi bridge does not support num_seasons > 1."
                )
            self.env = DssatPdi(**dssat_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct DssatPdi with params {env_params}: {e}"
            )
        finally:
            if orig_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = orig_pythonpath
            if orig_pythonhome is None:
                os.environ.pop("PYTHONHOME", None)
            else:
                os.environ["PYTHONHOME"] = orig_pythonhome

        # The worker is the DssatPdi instance
        self.worker = self.env

        # Store internal state pointer
        self._state = getattr(self.worker, "_state", None)

        # Multi-season tracking
        self._num_seasons = num_seasons
        self._current_season = 1  # 1-indexed, for display

        # Episode step counters (WOFOST-style tracking)
        self._total_steps = 0
        self._invalid_steps = 0

        # Cumulative input trackers (episode-level totals across all seasons)
        self._cumulative_fert = 0.0
        self._cumulative_irrig = 0.0
        self._fert_application_count = 0
        self._irrig_application_count = 0

        # Trajectory variable trackers
        self._max_xlai = 0.0
        self._final_grnwt = 0.0

        # Track current day after planting
        self._current_dap = 0

        # Decision interval: how many DSSAT days between LLM decisions
        self.decision_interval = getattr(config, "decision_interval", 1)

        # ====================================================================
        # PEST MANAGEMENT: Initialize pest simulation
        # ====================================================================
        self.enable_pests = getattr(config, "enable_pests", False)
        if self.enable_pests:
            self.pest_config = getattr(config, "pest_config", {})
            self._pest_pressure = 0.0  # Current pest pressure (0-1)
            self._cumulative_pest_damage = 0.0  # Cumulative yield loss from pests
            self._pesticide_applications = 0  # Number of pesticide applications
            self._days_since_pesticide = 999  # Days since last pesticide application
            print("✅ Pest management enabled")
        else:
            self.pest_config = {}
        # ====================================================================

        # ------------------------------------------------------------
        # Observation variables
        # ------------------------------------------------------------
        if hasattr(self.env, "get_output_vars"):
            try:
                self.output_vars = self.env.get_output_vars()
            except:
                self.output_vars = []
        else:
            self.output_vars = []

        # Fallback set (LLM should always get the same ordering)
        if not self.output_vars:
            self.output_vars = [
                "dap", "cumsumfert", "swfac", "vstage",
                "grnwt", "xlai", "tmax", "srad",
            ]

        # Add pest variables if enabled
        if self.enable_pests:
            pest_vars = ["pest_pressure", "pest_damage", "days_since_pesticide"]
            for pv in pest_vars:
                if pv not in self.output_vars:
                    self.output_vars.append(pv)

        # ------------------------------------------------------------
        # Prompt generator
        # ------------------------------------------------------------
        crop_name = getattr(config, "crop_name", None) or (config.env_params or {}).get("cultivar", "maize")
        self._crop_name_cached = crop_name
        self.objective_id = str(getattr(config, "objective_id", "profit_max") or "profit_max")
        self.reward_params = dict(getattr(config, "reward_params", {}) or {})
        objective_prompt_params = {
            **self.reward_params,
            **dict(getattr(config, "profit_context_params", {}) or {}),
        }
        _think_kwargs = dict(
            require_think=getattr(config, "require_think", False),
            include_crop_traits=getattr(config, "include_crop_traits", False),
            thinking_mode=getattr(config, "thinking_mode", "grounding_decision"),
            think_tag=getattr(config, "think_tag", "tool_call"),
            decision_interval=self.decision_interval,
            enable_pests=self.enable_pests,
            output_vars=self.output_vars,
            include_profit_context=getattr(config, "include_profit_context", False),
            profit_context_params=objective_prompt_params,
            objective_id=getattr(config, "prompt_objective_id", None) or self.objective_id,
            objective_text=getattr(config, "prompt_objective_text", None),
            reward_params=self.reward_params,
        )
        if crop_name == "cotton":
            from .prompt_cotton import CottonPromptGenerator
            self.prompt_generator = CottonPromptGenerator(**_think_kwargs)
        elif crop_name == "rice":
            from .prompt_rice import RicePromptGenerator
            self.prompt_generator = RicePromptGenerator(**_think_kwargs)
        else:
            self.prompt_generator = DSSATPromptGenerator(crop_name=crop_name, **_think_kwargs)
        self.llm_mode = getattr(config, "llm_mode", True)

        # Reward key default
        self.reward_key = getattr(config, "reward_key", "yield")
        reward_mode = str(getattr(config, "env_reward", "") or "").lower()
        self._yield_only_terminal_reward = reward_mode in {
            "yield_only",
            "yield_only_terminal",
            "sparse_terminal_yield",
            "rewardfinalyieldwrapper",
        } or self.objective_id == "yield_max"
        self._profit_terminal_reward = self.objective_id == "profit_max"
        self.valid_action_bonus = float(getattr(config, "valid_action_bonus", 0.1))

        super().__init__(config)

        print("DEBUG: worker type =", type(self.worker))

    @property
    def unwrapped(self):
        """Gym-style compatibility: return the base environment."""
        return self
    def parse_action_response(self, response: str):
        """Compatibility layer: forward to prompt generator."""
        return self.prompt_generator.parse_action_response(response)
    # -------------------------------------------------------------------------
    # Map discrete action id → physical fertilizer/irrigation/pesticide amounts
    # -------------------------------------------------------------------------
    def _id_to_action_dict(self, action_id: int) -> Dict[str, float]:

        num_fert = self.prompt_generator.num_fert
        num_irrig = self.prompt_generator.num_irrig
        fert_step = self.prompt_generator.fert_amount
        irrig_step = self.prompt_generator.irrig_amount

        # 0 → do nothing
        if action_id == 0:
            return {"anfer": 0.0, "amir": 0.0, "pesticide": 0.0}

        # Fertilizer actions
        if 1 <= action_id <= num_fert:
            amt = action_id * fert_step
            return {"anfer": float(amt), "amir": 0.0, "pesticide": 0.0}

        # Irrigation actions
        irr_id = action_id - num_fert
        if 1 <= irr_id <= num_irrig:
            amt = irr_id * irrig_step
            return {"anfer": 0.0, "amir": float(amt), "pesticide": 0.0}

        # Pesticide action (if enabled)
        if self.enable_pests:
            pest_id = action_id - num_fert - num_irrig
            if pest_id == 1:
                return {"anfer": 0.0, "amir": 0.0, "pesticide": 1.0}

        # Fallback
        return {"anfer": 0.0, "amir": 0.0, "pesticide": 0.0}

    def _clean_action_dict(self, action: Dict[str, float]) -> Dict[str, float]:
        """Normalize DSSAT action dictionaries from parser or numeric adapters."""
        return {
            "anfer": float(action.get("anfer", 0.0)),
            "amir": float(action.get("amir", 0.0)),
            "pesticide": float(action.get("pesticide", 0.0)) if self.enable_pests else 0.0,
        }

    def _profit_diagnostics(self) -> Dict[str, float]:
        costs = profit_cost_params(self.reward_params)
        revenue = float(self._final_grnwt)
        nitrogen_cost = costs["cost_n"] * float(self._cumulative_fert)
        irrigation_cost = costs["cost_water"] * float(self._cumulative_irrig)
        input_cost = nitrogen_cost + irrigation_cost
        profit = revenue - input_cost
        scale = profit_reward_scale(self.reward_params)
        return {
            "objective_reward": float(profit / scale),
            "reward_profit_term": float(profit / scale),
            "revenue_ge_kg_ha": revenue,
            "input_cost_ge_kg_ha": float(input_cost),
            "nutrient_cost_ge_kg_ha": float(nitrogen_cost),
            "irrigation_cost_ge_kg_ha": float(irrigation_cost),
            "profit_ge_kg_ha": float(profit),
        }

    def _terminal_objective_reward(self, done: bool, native_reward: float) -> tuple[float, Dict[str, float]]:
        if getattr(self, "_profit_terminal_reward", False):
            if not done:
                return 0.0, {}
            diagnostics = self._profit_diagnostics()
            return diagnostics["objective_reward"], diagnostics
        if getattr(self, "_yield_only_terminal_reward", False):
            return float(self._final_grnwt / 1000.0 if done else 0.0), {}
        return float(native_reward), {}

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def reset(self) -> Tuple[Any, Dict[str, Any]]:

        raw = self.worker.reset()
        obs = raw

        # Reset all episode trackers
        self._total_steps = 0
        self._invalid_steps = 0
        self._cumulative_fert = 0.0
        self._cumulative_irrig = 0.0
        self._fert_application_count = 0
        self._irrig_application_count = 0
        self._max_xlai = 0.0
        self._final_grnwt = 0.0
        self._current_dap = 0
        self._current_season = 1

        # Reset pest trackers
        if self.enable_pests:
            self._pest_pressure = self.pest_config.get("base_pressure", 0.3)
            self._cumulative_pest_damage = 0.0
            self._pesticide_applications = 0
            self._days_since_pesticide = 999

        current_lai = self._extract_obs_value(obs, "lai" if self._crop_name_cached == "rice" else "xlai")
        current_grnwt = self._extract_obs_value(obs, "grnwt")
        info = {
            "metrics": {"reward": 0.0},
            "turn_metrics": {
                "dap": self._extract_obs_value(obs, "dap"),
                "grnwt": current_grnwt,
                "xlai": current_lai,
                "topwt": self._extract_obs_value(obs, "topwt"),
                "nstres": self._extract_obs_value(obs, "nstres"),
                "swfac": self._extract_obs_value(obs, "swfac"),
                "cumsumfert": 0.0,
                "cumsumirrg": 0.0,
                "anfer_applied": 0.0,
                "amir_applied": 0.0,
                "action_type": "reset",
                "reward": 0.0,
            },
        }

        # Augment observation with pest variables if enabled
        if self.enable_pests:
            obs = self._add_pest_to_observation(obs)

        if self.llm_mode:
            turn_prompt = self.prompt_generator.get_turn_prompt(
                obs, season_num=self._current_season, num_seasons=self._num_seasons
            )
            return turn_prompt, info

        return obs, info

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action: Union[str, int, float, Dict[str, float]]):

        raw_llm_response = None
        invalid_action = False
        valid_action_format = False

        # Convert LLM string → discrete action id → action dict
        if isinstance(action, str):
            raw_llm_response = action
            parsed_action = self.prompt_generator.parse_action_response(action)
            if parsed_action is None:
                parsed_action = 0  # unparseable → do nothing
                invalid_action = True
            else:
                valid_action_format = True
            action_val = (
                self._clean_action_dict(parsed_action)
                if isinstance(parsed_action, dict)
                else self._id_to_action_dict(parsed_action)
            )

        # Dict: already clean {anfer, amir, pesticide?}
        elif isinstance(action, dict):
            action_val = self._clean_action_dict(action)

        # Numeric: treat as discrete id
        else:
            action_id = int(action)
            action_val = self._id_to_action_dict(action_id)

        # Track invalid actions
        self._total_steps += 1
        if invalid_action:
            self._invalid_steps += 1

        # Track fertilizer/irrigation inputs
        if action_val.get('anfer', 0) > 0:
            self._cumulative_fert += action_val['anfer']
            self._fert_application_count += 1
        if action_val.get('amir', 0) > 0:
            self._cumulative_irrig += action_val['amir']
            self._irrig_application_count += 1

        # ====================================================================
        # PEST MANAGEMENT: Handle pesticide application
        # ====================================================================
        if self.enable_pests and action_val.get('pesticide', 0) > 0:
            self._pesticide_applications += 1
            self._days_since_pesticide = 0
            efficacy = self.pest_config.get("pesticide_efficacy", 0.7)
            self._pest_pressure *= (1 - efficacy)
            if self.llm_mode:
                print(f"✅ Pesticide applied! Pest pressure reduced by {efficacy*100:.0f}%")
        # ====================================================================

        # Always send CLEAN action dict to DSSAT (no pesticide key)
        dssat_action = {"anfer": action_val["anfer"], "amir": action_val["amir"]}

        # ====================================================================
        # DECISION INTERVAL: step DSSAT N days for one LLM decision
        # ====================================================================
        reward = 0.0
        obs_raw = None
        done = False
        info_raw = {}
        for day_i in range(self.decision_interval):
            # Only apply action on first day of the interval; do nothing on rest
            step_action = dssat_action if day_i == 0 else {"anfer": 0.0, "amir": 0.0}
            out = self.worker.step(step_action)
            if not isinstance(out, tuple) or len(out) != 4:
                raise ValueError(f"Unexpected worker.step output format: {out}")
            obs_raw, reward_raw, done, info_raw = out
            try:
                reward += float(sum(reward_raw)) if isinstance(reward_raw, (list, tuple)) else float(reward_raw)
            except:
                pass
            if done:
                break
        # ====================================================================

        self._state = getattr(self.worker, "_state", None)

        # Detect season boundary: dssat_pdi._season_num is 0-indexed.
        # Only update when obs_raw is valid (skip the terminal done=True with obs_raw=None).
        new_season = getattr(self.worker, "_season_num", 0) + 1  # convert to 1-indexed
        if new_season > self._current_season and obs_raw is not None:
            # Season boundary crossed — reset per-season trackers, keep episode totals
            self._current_season = new_season
            self._fert_application_count = 0
            self._irrig_application_count = 0
            self._max_xlai = 0.0
            self._final_grnwt = 0.0
            self._current_dap = 0

        # Extract DAP and plant variables from last observation
        self._current_dap = int(self._extract_obs_value(obs_raw, 'dap'))
        # Rice obs uses 'lai' instead of 'xlai' (CERES-Rice variable naming);
        # cotton and maize use 'xlai'.
        _lai_key = "lai" if self._crop_name_cached == "rice" else "xlai"
        current_xlai = self._extract_obs_value(obs_raw, _lai_key)
        current_grnwt = self._extract_obs_value(obs_raw, 'grnwt')
        if current_xlai > self._max_xlai:
            self._max_xlai = current_xlai
        if current_grnwt > 0:
            self._final_grnwt = current_grnwt

        # PEST MANAGEMENT: Update pest pressure and damage
        if self.enable_pests:
            self._update_pest_pressure(obs_raw)
            obs_raw = self._add_pest_to_observation(obs_raw)

        # Apply pest damage penalty to reward
        if self.enable_pests:
            pest_cost = self._pesticide_applications * self.pest_config.get("pesticide_cost", 15.0)
            pest_damage_penalty = self._cumulative_pest_damage
            reward = reward - pest_cost - pest_damage_penalty

        reward, reward_diagnostics = self._terminal_objective_reward(bool(done), float(reward))
        if valid_action_format:
            reward += self.valid_action_bonus

        # ====================================================================
        # Build per-turn metrics (WOFOST-style)
        # ====================================================================
        action_type = "nothing"
        if action_val.get('anfer', 0) > 0 and action_val.get('amir', 0) > 0:
            action_type = "fert+irrig"
        elif action_val.get('anfer', 0) > 0:
            action_type = "fertilize"
        elif action_val.get('amir', 0) > 0:
            action_type = "irrigate"

        turn_metrics = {
            "dap": self._current_dap,
            "grnwt": current_grnwt,
            "xlai": current_xlai,
            "topwt": self._extract_obs_value(obs_raw, 'topwt'),
            "nstres": self._extract_obs_value(obs_raw, 'nstres'),
            "swfac": self._extract_obs_value(obs_raw, 'swfac'),
            "cumsumfert": self._cumulative_fert,
            "cumsumirrg": self._cumulative_irrig,
            "anfer_applied": float(action_val.get("anfer", 0.0)),
            "amir_applied": float(action_val.get("amir", 0.0)),
            "action_type": action_type,
            "reward": reward,
        }
        turn_metrics.update(reward_diagnostics)

        # LLM mode → return prompt instead of raw obs
        if self.llm_mode:
            if obs_raw is None or bool(done):
                turn_prompt = "Episode complete."
            else:
                turn_prompt = self.prompt_generator.get_turn_prompt(
                    obs_raw, season_num=self._current_season, num_seasons=self._num_seasons
                )

            info = {
                "metrics": {"reward": reward},
                "turn_metrics": turn_metrics,
            }
            if raw_llm_response is not None:
                info["raw_llm_response"] = raw_llm_response
            info["action_applied"] = {
                "anfer": float(action_val.get("anfer", 0.0)),
                "amir": float(action_val.get("amir", 0.0))
            }

            # Trajectory metrics at episode end (WOFOST-style)
            if done:
                info["trajectory_metrics"] = self.get_trajectory_metrics()

            return turn_prompt, reward, bool(done), info

        return obs_raw, reward, bool(done), info_raw

    # -------------------------------------------------------------------------
    # SYSTEM PROMPT
    # -------------------------------------------------------------------------
    def system_prompt(self) -> str:
        return self.prompt_generator.get_system_prompt()

    def get_trajectory_metrics(self) -> Dict[str, float]:
        total = max(self._total_steps, 1)
        metrics = {
            "yield_kgha": float(self._final_grnwt),
            "target_yield": float(self._final_grnwt),
            "max_xlai": float(self._max_xlai),
            "total_fert": float(self._cumulative_fert),
            "total_irrig": float(self._cumulative_irrig),
            "total_n_kg_ha": float(self._cumulative_fert),
            "total_irrig_mm": float(self._cumulative_irrig),
            "fert_applications": float(self._fert_application_count),
            "irrig_applications": float(self._irrig_application_count),
            "invalid_action_rate": float(self._invalid_steps / total),
            "invalid_steps": float(self._invalid_steps),
            "total_steps": float(self._total_steps),
        }
        if getattr(self, "_profit_terminal_reward", False):
            metrics.update(self._profit_diagnostics())
        return metrics

    # -------------------------------------------------------------------------
    # PEST MANAGEMENT HELPERS
    # -------------------------------------------------------------------------
    def _extract_obs_value(self, obs, key: str) -> float:
        """Read one observation value from dict- or array-like observations."""
        if isinstance(obs, dict):
            return float(obs.get(key, 0.0))
        if hasattr(obs, "__len__") and key in self.output_vars:
            try:
                return float(obs[self.output_vars.index(key)])
            except Exception:
                return 0.0
        return 0.0

    def _update_pest_pressure(self, obs):
        """Update pest pressure based on weather and time."""
        # Increment days since pesticide
        self._days_since_pesticide += 1

        # Extract weather variables
        if isinstance(obs, dict):
            tmax = obs.get('tmax', 25)
            srad = obs.get('srad', 20)
        elif hasattr(obs, '__len__') and 'tmax' in self.output_vars:
            try:
                tmax_idx = self.output_vars.index('tmax')
                tmax = obs[tmax_idx]
            except:
                tmax = 25
            try:
                srad_idx = self.output_vars.index('srad')
                srad = obs[srad_idx]
            except:
                srad = 20
        else:
            tmax = 25
            srad = 20

        # Pest pressure increases with warm weather and solar radiation
        weather_sensitivity = self.pest_config.get("weather_sensitivity", 0.5)
        temp_factor = max(0, (tmax - 20) / 15)  # Increases above 20°C
        light_factor = max(0, (srad - 15) / 10)  # Increases with sunlight

        # Natural growth rate
        growth_rate = 0.05 * (1 + weather_sensitivity * (temp_factor + light_factor) / 2)

        # Pesticide decay effect
        decay_from_pesticide = 0.0
        if self._days_since_pesticide < 14:  # Pesticide effective for ~2 weeks
            decay_from_pesticide = 0.02 * (14 - self._days_since_pesticide) / 14

        # Update pressure (bounded 0-1)
        self._pest_pressure += growth_rate - decay_from_pesticide
        self._pest_pressure = max(0.0, min(1.0, self._pest_pressure))

        # Accumulate damage
        damage_rate = self.pest_config.get("damage_rate", 0.02)
        daily_damage = damage_rate * self._pest_pressure
        self._cumulative_pest_damage += daily_damage

    def _add_pest_to_observation(self, obs):
        """Add pest variables to observation."""
        if isinstance(obs, dict):
            obs['pest_pressure'] = self._pest_pressure
            obs['pest_damage'] = self._cumulative_pest_damage
            obs['days_since_pesticide'] = self._days_since_pesticide
            return obs
        elif hasattr(obs, '__len__'):
            # Convert to list, add pest vars, return as array
            import numpy as np
            obs_list = list(obs) if not isinstance(obs, list) else obs
            obs_list.extend([
                self._pest_pressure,
                self._cumulative_pest_damage,
                self._days_since_pesticide
            ])
            return np.array(obs_list) if isinstance(obs, np.ndarray) else obs_list
        else:
            return obs

    # -------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------
    def close(self) -> None:
        if self.worker:
            self.worker.close()
