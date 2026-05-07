"""Dataset generation for gym-dssat environments (seed-based, crop-agnostic)."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from agrimanager.env.base import BaseDatasetGenerator
from agrimanager.env.gym_dssat.env_config import (
    DEFAULT_DSSAT_GYM_PATH,
    DEFAULT_DSSAT_OUTPUT_PATH,
)

# Project root: agrimanager/env/gym_dssat/create_dataset.py → 3 levels up
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Standard 8-layer soil depths (shared across all crops)
_LAYER_DEPTHS = [5, 15, 30, 60, 90, 120, 150, 180]


# ---------------------------------------------------------------------------
# Soil generation helpers
# ---------------------------------------------------------------------------

def _get_soil_dir(dssat_gym_path: str) -> Path:
    """Resolve the DSSAT Soil directory from the gym-dssat installation path.

    Can be overridden at runtime via the DSSAT_SOIL_DIR environment variable.
    """
    env_override = os.environ.get("DSSAT_SOIL_DIR")
    if env_override:
        return Path(env_override)
    return Path(dssat_gym_path.rstrip("/\\")) / "bin" / "Soil"


def _random_soil_profile(
    soil_id: str,
    rng: np.random.Generator,
    soil_label: str = "Custom",
    soil_params: Dict[str, Any] | None = None,
) -> str:
    """Return a DSSAT SOL-format string for one randomly generated soil profile.

    Constraints kept physically realistic:
      SLLL < SDUL < SSAT  (wilting pt < field capacity < saturation)
      SRGF decays from 1.0 at surface toward 0 at depth
      SLOC (organic C) decays exponentially with depth

    Args:
        soil_id: DSSAT soil identifier (≤10 chars).
        rng: NumPy random generator for reproducibility.
        soil_label: Human-readable label written into the SOL header.
        soil_params: Optional dict overriding default sampling ranges.
            Supported keys (all optional):
                salb_range, slu1_range, sldr_range, slro_range,
                slpf_range, base_clay_range
            Each value is a [low, high] list.
    """
    sp = soil_params or {}

    def _rng_range(key: str, default_low: float, default_high: float) -> float:
        lo, hi = sp.get(key, [default_low, default_high])
        return float(rng.uniform(lo, hi))

    salb      = _rng_range("salb_range",      0.10, 0.22)
    slu1      = _rng_range("slu1_range",       2.0,  9.0)
    sldr      = _rng_range("sldr_range",      0.30, 0.80)
    slro      = _rng_range("slro_range",      60.0, 85.0)
    slpf      = _rng_range("slpf_range",      0.75, 1.00)
    base_clay = _rng_range("base_clay_range", 0.03, 0.45)

    n = len(_LAYER_DEPTHS)
    layer_lines = []
    for i, slb in enumerate(_LAYER_DEPTHS):
        frac = i / max(n - 1, 1)
        clay = float(np.clip(base_clay + rng.uniform(-0.04, 0.06) * frac, 0.02, 0.55))
        slll = float(np.clip(0.03 + 0.35 * clay + rng.uniform(-0.02, 0.02), 0.02, 0.38))
        sdul = float(np.clip(slll + 0.06 + 0.25 * clay + rng.uniform(-0.02, 0.02), slll + 0.03, 0.50))
        ssat = float(np.clip(sdul + 0.05 + rng.uniform(0.00, 0.10), sdul + 0.02, 0.60))
        srgf = float(max(0.001, 1.0 - frac ** 0.7))
        sbdm = float(np.clip(2.65 * (1.0 - ssat) + rng.uniform(-0.05, 0.05), 1.00, 1.90))
        sloc = float(np.clip(2.5 * np.exp(-3.0 * frac) + rng.uniform(-0.10, 0.10), 0.01, 4.00))
        layer_lines.append(
            f"  {slb:4d}   -99 {slll:.3f} {sdul:.3f} {ssat:.3f} {srgf:.3f}"
            f"   -99  {sbdm:.2f}  {sloc:.2f}   -99   -99   -99   -99   -99   -99   -99   -99"
        )

    lines = [
        f"*{soil_id:<10}  CUSTOM      -99     180 Random {soil_label} Soil",
        f"@SITE        COUNTRY          LAT     LONG SCS FAMILY",
        f" Custom      Custom           -99      -99 Custom Random Profile",
        f"@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE",
        f"   -99  {salb:.2f}  {slu1:.1f}  {sldr:.2f}  {slro:.1f}  1.00  {slpf:.2f} IB001 IB001 IB001",
        f"@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC",
        *layer_lines,
        "",
    ]
    return "\n".join(lines)


def _write_custom_sol(
    profiles: Dict[str, str],
    soil_dir: Path,
    sol_file: str,
    soil_label: str = "crop",
) -> None:
    """Write all generated profiles to a .SOL file in DSSAT's Soil directory."""
    out = soil_dir / sol_file
    with open(out, "w") as f:
        f.write(f"$SOILS: Randomly generated {soil_label} profiles for RL training diversity\n\n")
        for text in profiles.values():
            f.write(text + "\n")


def _make_filex(
    soil_id: str,
    filex_dir: Path,
    base_filex: Path,
    template_soil_id: str,
) -> str:
    """Copy base fileX template with soil ID substituted; return the file path."""
    content = base_filex.read_text()
    content = content.replace(template_soil_id, soil_id)
    path = filex_dir / f"{soil_id}.jinja2"
    path.write_text(content)
    return str(path)


def _resolve_seeds(seeds_cfg: Union[list, dict], label: str) -> List[int]:
    """Return a flat list of integer seeds from either a list or a count dict."""
    if isinstance(seeds_cfg, list):
        return list(seeds_cfg)
    if isinstance(seeds_cfg, dict):
        count = seeds_cfg.get("count", 1)
        base = seeds_cfg.get("base_seed", 0)
        return list(range(base, base + count))
    raise ValueError(f"'{label}' must be a list or a dict with 'count'/'base_seed'.")


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class DSSATDatasetGenerator(BaseDatasetGenerator):
    """Generate train/test splits for gym-dssat across any supported crop.

    Crop-specific soil configuration is read from the YAML config under
    ``soil_variation``:

    .. code-block:: yaml

        soil_variation:
          enabled: true
          master_seed: 42
          sol_file: "CS.SOL"           # DSSAT .SOL filename (2-char prefix of soil_id_prefix)
          soil_id_prefix: "CST"        # Prefix for generated soil IDs (≤3 chars)
          base_filex: "path/to/template.jinja2"  # Absolute or PROJECT_ROOT-relative
          template_soil_id: "IBMZ910014"  # Token to replace in the fileX template
          soil_label: "Maize"          # Human-readable label for the SOL header
          soil_params:                 # Optional: override default sampling ranges
            salb_range: [0.10, 0.22]
            sldr_range: [0.30, 0.80]
            slro_range: [60.0, 85.0]
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self.base_config = self._build_base_config()
        self._soil_map: Dict[int, str] = {}   # seed → fileX_template_path
        self._init_soil_variants()

    def _build_base_config(self) -> Dict[str, Any]:
        # Env var takes precedence over YAML value, which takes precedence over code default.
        dssat_gym_path = (
            os.environ.get("DSSAT_GYM_PATH")
            or self.config.get("dssat_gym_path")
            or DEFAULT_DSSAT_GYM_PATH
        )
        base = {
            "env_name":      self.config.get("env_name", "gym_dssat"),
            "env_id":        self.config.get("env_id", "maize-irrigation-v0"),
            "dssat_gym_path": dssat_gym_path,
            "save_folder":   self.config.get("save_folder", DEFAULT_DSSAT_OUTPUT_PATH),
            "llm_mode":      self.config.get("llm_mode", True),
            "crop_name":     self.config.get("crop_name", "maize"),
            "env_params":    deepcopy(self.config.get("env_params", {})),
            "turn_num":      self.config.get("turn_num", 200),
            "decision_interval": self.config.get("decision_interval", 1),
            "num_seasons":   self.config.get("num_seasons", 1),
            "enable_pests":  self.config.get("enable_pests", False),
            "require_think": self.config.get("require_think", False),
            "include_crop_traits": self.config.get("include_crop_traits", False),
            "thinking_mode": self.config.get("thinking_mode", "grounding_decision"),
            "think_tag":     self.config.get("think_tag", "tool_call"),
            "objective_id":   self.config.get("objective_id", "profit_max"),
            "include_profit_context": self.config.get("include_profit_context", False),
        }
        if self.config.get("env_reward"):
            base["env_reward"] = self.config["env_reward"]
        if self.config.get("prompt_objective_id"):
            base["prompt_objective_id"] = self.config["prompt_objective_id"]
        if self.config.get("prompt_objective_text"):
            base["prompt_objective_text"] = self.config["prompt_objective_text"]
        if self.config.get("reward_params"):
            base["reward_params"] = deepcopy(self.config["reward_params"])
        if self.config.get("trajectory_group_labels"):
            base["trajectory_group_labels"] = deepcopy(
                self.config["trajectory_group_labels"]
            )
        profit_context_params = self.config.get("profit_context_params")
        if profit_context_params:
            base["profit_context_params"] = deepcopy(profit_context_params)
        dssat_params = self.config.get("dssat_params", {})
        if dssat_params:
            base["dssat_params"] = deepcopy(dssat_params)
        pest_config = self.config.get("pest_config")
        if pest_config:
            base["pest_config"] = deepcopy(pest_config)
        return base

    def _init_soil_variants(self) -> None:
        """If soil_variation is enabled, generate profiles for every seed."""
        sv = self.config.get("soil_variation", {})
        if not sv.get("enabled", False):
            return

        # ── resolve crop-specific soil config from YAML ──────────────────────
        dssat_gym_path = self.base_config["dssat_gym_path"]
        soil_dir       = _get_soil_dir(dssat_gym_path)
        sol_file       = sv.get("sol_file", "CS.SOL")
        soil_id_prefix = sv.get("soil_id_prefix", "CST")
        soil_label     = sv.get("soil_label", "Custom")
        soil_params    = sv.get("soil_params", {})
        template_soil_id = sv.get("template_soil_id", "IBMZ910014")

        raw_filex = sv.get("base_filex")
        if not raw_filex:
            raise ValueError(
                "soil_variation.base_filex must be set in the dataset config. "
                "Provide an absolute path, a path relative to dssat_gym_path, "
                "or a path relative to the project root (parent of dssat_gym_path)."
            )
        base_filex = Path(raw_filex)
        if not base_filex.is_absolute():
            # Resolution order:
            # 1. Relative to project root (where AgriManagerExternal lives)
            # 2. Relative to dssat_gym_path
            for base in (_PROJECT_ROOT, Path(dssat_gym_path.rstrip("/\\"))):
                candidate = base / raw_filex
                if candidate.exists():
                    base_filex = candidate
                    break
        if not base_filex.exists():
            raise FileNotFoundError(
                f"base_filex not found: {base_filex}\n"
                f"  Tried relative to project root: {_PROJECT_ROOT / raw_filex}\n"
                f"  Tried relative to dssat_gym_path: {Path(dssat_gym_path) / raw_filex}"
            )

        master_seed = sv.get("master_seed", 42)
        rng = np.random.default_rng(master_seed)

        all_seeds: List[int] = []
        for split in ("train", "val", "test"):
            cfg = self.config.get(f"{split}_seeds")
            if cfg:
                all_seeds.extend(_resolve_seeds(cfg, f"{split}_seeds"))

        if not all_seeds:
            return

        filex_dir = Path(self.output_dir) / "filex_templates"
        filex_dir.mkdir(parents=True, exist_ok=True)

        profiles: Dict[str, str] = {}
        for seed in all_seeds:
            soil_id = f"{soil_id_prefix}{seed:07d}"[:10]
            profiles[soil_id] = _random_soil_profile(
                soil_id, rng, soil_label=soil_label, soil_params=soil_params
            )
            self._soil_map[seed] = _make_filex(
                soil_id, filex_dir, base_filex, template_soil_id
            )

        _write_custom_sol(profiles, soil_dir, sol_file, soil_label)
        print(
            f"[soil] Generated {len(profiles)} random {soil_label} soil profiles "
            f"→ {soil_dir / sol_file}"
        )

    def splits(self) -> List[str]:
        splits = []
        for split in ("train", "val", "test"):
            if self.config.get(f"{split}_seeds"):
                splits.append(split)
        if not splits:
            raise ValueError(
                "Config must specify at least one of 'train_seeds', 'val_seeds', or 'test_seeds'."
            )
        return splits

    def build_split_records(self, split: str) -> List[Dict[str, Any]]:
        seeds_key = f"{split}_seeds"
        seeds_cfg = self.config.get(seeds_key)
        if not seeds_cfg:
            raise ValueError(f"Config must specify '{seeds_key}'.")

        seeds = _resolve_seeds(seeds_cfg, seeds_key)

        records = []
        for row_index, seed in enumerate(seeds):
            cfg = deepcopy(self.base_config)
            cfg["seed"] = seed
            cfg["env_params"] = deepcopy(self.base_config["env_params"])
            cfg["env_params"]["seed"] = seed
            crop_name = str(cfg.get("crop_name") or cfg["env_params"].get("cultivar", "maize") or "maize")
            cfg["crop_name"] = crop_name
            cfg["dataset_id"] = self.dataset_id
            cfg["dataset_split"] = split
            cfg["scenario_id"] = self._scenario_id(split, row_index, cfg)
            group_labels = dict(cfg.get("trajectory_group_labels") or {})
            group_labels.setdefault("simulator", cfg.get("env_name", "gym_dssat"))
            group_labels.setdefault("split", split)
            group_labels.setdefault("crop", crop_name)
            group_labels.setdefault("objective_id", str(cfg.get("objective_id", "profit_max")))
            cfg["trajectory_group_labels"] = group_labels
            if seed in self._soil_map:
                cfg["env_params"]["fileX_template_path"] = self._soil_map[seed]
            records.append(cfg)

        return records

    def _scenario_id(self, split: str, row_index: int, cfg: Dict[str, Any]) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "split": split,
            "row_index": row_index,
            "seed": cfg.get("seed"),
            "env_id": cfg.get("env_id"),
            "env_params": cfg.get("env_params", {}),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"dssat_{digest[:16]}"

    def _generate_more_records(self, split: str, count: int) -> List[Dict[str, Any]]:
        """Generate replacement records by extending the split seed sequence."""
        seeds_key = f"{split}_seeds"
        seeds_cfg = self.config.get(seeds_key)
        if not isinstance(seeds_cfg, dict):
            raise ValueError(
                f"Automatic retry for '{split}' requires {seeds_key} to be a dict "
                "with 'count' and 'base_seed'."
            )

        base_seed = int(seeds_cfg.get("base_seed", 0))
        original_count = int(seeds_cfg.get("count", 0))
        retry_start = base_seed + original_count

        records = []
        for offset in range(count):
            seed = retry_start + offset
            cfg = deepcopy(self.base_config)
            cfg["seed"] = seed
            cfg["env_params"] = deepcopy(self.base_config["env_params"])
            cfg["env_params"]["seed"] = seed
            crop_name = str(cfg.get("crop_name") or cfg["env_params"].get("cultivar", "maize") or "maize")
            cfg["crop_name"] = crop_name
            cfg["dataset_id"] = self.dataset_id
            cfg["dataset_split"] = split
            cfg["scenario_id"] = self._scenario_id(split, original_count + offset, cfg)
            group_labels = dict(cfg.get("trajectory_group_labels") or {})
            group_labels.setdefault("simulator", cfg.get("env_name", "gym_dssat"))
            group_labels.setdefault("split", split)
            group_labels.setdefault("crop", crop_name)
            group_labels.setdefault("objective_id", str(cfg.get("objective_id", "profit_max")))
            cfg["trajectory_group_labels"] = group_labels
            if seed in self._soil_map:
                cfg["env_params"]["fileX_template_path"] = self._soil_map[seed]
            records.append(cfg)
        return records

    def _placeholder_prompt_env_label(self, env_config: Dict[str, Any]) -> str:
        return "Gym-DSSAT"

    def _placeholder_prompt_context(self, env_config: Dict[str, Any]) -> str:
        env_params = env_config.get("env_params") or {}
        crop = env_params.get("cultivar", "maize")
        seed = env_params.get("seed", env_config.get("seed"))
        random_weather = env_params.get("random_weather", False)
        weather = "random weather" if random_weather else "fixed weather"
        return f"{crop}, {weather}, seed={seed}"


def generate(config: Dict[str, Any], output_dir: str):
    """Entry point used by agrimanager.env.create_dataset router."""
    builder = DSSATDatasetGenerator(config, output_dir)
    builder.generate()
