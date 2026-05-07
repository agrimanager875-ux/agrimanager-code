"""Dataset generation for wofost_gym environments."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from collections import Counter
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from agrimanager.env.base import BaseDatasetGenerator
from agrimanager.env.base.create_dataset import _worker_convert
from agrimanager.env.base.dataset_metadata import (
    apply_split_metadata_to_env_config,
    normalize_split_metadata,
)
from agrimanager.env.base.utils import create_environment
from agrimanager.env.wofost_gym.env_config import DEFAULT_WOFOST_GYM_PATH, WOFOSTEnvConfig
from agrimanager.env.wofost_gym.weather_pool import (
    ensure_pool,
    find_pool_meteo_cache_dir,
    load_pool,
    sample_scenarios,
)


def _stable_u32_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def scenario_id_for_row(
    split: str,
    crop_name: str,
    year: int,
    latitude: float,
    longitude: float,
) -> str:
    payload = f"{split}|{crop_name}|{year}|{latitude:.2f}|{longitude:.2f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def scenario_id_for_variety_row(
    split: str,
    crop_name: str,
    crop_variety: str,
    year: int,
    latitude: float,
    longitude: float,
) -> str:
    payload = (
        f"{split}|{crop_name}|{crop_variety}|"
        f"{year}|{latitude:.2f}|{longitude:.2f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def paired_scenario_id_for_row(
    crop_name: str,
    year: int,
    latitude: float,
    longitude: float,
    crop_variety: str | None = None,
) -> str:
    """Split-independent scenario identifier for paired schema comparisons."""
    payload = f"{crop_name}|{year}|{latitude:.2f}|{longitude:.2f}"
    if crop_variety:
        payload = f"{crop_name}|{crop_variety}|{year}|{latitude:.2f}|{longitude:.2f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def env_seed_from_scenario_id(scenario_id: str) -> int:
    return int(scenario_id[:8], 16)


def crop_sampling_seed(split_seed: int, crop_name: str) -> int:
    """Derive a deterministic per-crop RNG seed for artifact sampling.

    This decouples one crop's sampled rows from every other crop in the same
    split, so shared crops with the same budget get the same sampled weather
    rows across different dataset compositions.
    """
    return _stable_u32_seed("wofost_crop_sampling", split_seed, crop_name)


_WEATHER_REGIME_SPLIT_LABELS = {
    "val_id": "id",
    "val_drought": "drought",
    "val_wet": "wet",
    "val_hot": "hot",
    "val_cold": "cold",
}


def _split_metadata(
    split: str,
    split_cfg: Dict[str, Any] | None = None,
    dataset_labels: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    inferred_labels: Dict[str, str] = {}
    weather_regime = _WEATHER_REGIME_SPLIT_LABELS.get(split)
    if weather_regime:
        inferred_labels["weather_regime"] = weather_regime
    return normalize_split_metadata(
        split,
        split_cfg or {},
        dataset_labels=dataset_labels,
        inferred_labels=inferred_labels,
    )


def _split_group_labels(
    split: str,
    split_cfg: Dict[str, Any] | None = None,
    dataset_labels: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    return dict(_split_metadata(split, split_cfg, dataset_labels)["group_labels"])


def _normalize_scenario_slice(raw_slice: Any) -> Dict[str, int] | None:
    if raw_slice in (None, "", {}):
        return None
    if not isinstance(raw_slice, dict):
        raise ValueError("scenario_slice must be a mapping when provided.")
    normalized: Dict[str, int] = {}
    for key in ("start", "stop", "step"):
        if key in raw_slice and raw_slice[key] is not None:
            normalized[key] = int(raw_slice[key])
    if "stop" not in normalized:
        raise ValueError("scenario_slice must define at least 'stop'.")
    normalized.setdefault("start", 0)
    normalized.setdefault("step", 1)
    if normalized["step"] <= 0:
        raise ValueError("scenario_slice.step must be positive.")
    if normalized["start"] < 0 or normalized["stop"] < 0:
        raise ValueError("scenario_slice bounds must be non-negative.")
    if normalized["stop"] < normalized["start"]:
        raise ValueError("scenario_slice.stop must be >= scenario_slice.start.")
    return normalized


def _sync_objective_group_label(env_config: Dict[str, Any]) -> Dict[str, Any]:
    objective_id = str(env_config.get("objective_id", "") or "").strip()
    if objective_id:
        group_labels = dict(env_config.get("trajectory_group_labels") or {})
        group_labels["objective_id"] = objective_id
        env_config["trajectory_group_labels"] = group_labels
    return env_config


def _drop_implicit_save_folder(
    env_config: Dict[str, Any],
    config_source: Dict[str, Any],
) -> Dict[str, Any]:
    """Omit implicit optional fields unless the dataset config explicitly set them."""
    if "save_folder" not in config_source or env_config.get("save_folder") is None:
        env_config.pop("save_folder", None)
    if "env_reward" not in config_source or not env_config.get("env_reward"):
        env_config.pop("env_reward", None)
    if "reward_params" not in config_source or not env_config.get("reward_params"):
        env_config.pop("reward_params", None)
    if "profit_context_params" not in config_source or not env_config.get("profit_context_params"):
        env_config.pop("profit_context_params", None)
    return env_config


def _infer_crop_name(env_config: Dict[str, Any]) -> str:
    crop_name = env_config.get("crop_name")
    if crop_name:
        return str(crop_name)

    agro_file = str(env_config.get("agro_file", "") or "")
    if agro_file.endswith("_agro.yaml"):
        return agro_file[:-10]
    if agro_file.endswith(".yaml"):
        return agro_file[:-5]
    return agro_file or "unknown_crop"


def _describe_placeholder_context(env_config: Dict[str, Any]) -> str:
    agro_params = env_config.get("agro_params") or {}
    crop_name = _infer_crop_name(env_config)
    crop_variety = ""
    agro_params = env_config.get("agro_params") or {}
    if isinstance(agro_params, dict):
        crop_variety = str(agro_params.get("crop_variety", "") or "").strip()
    crop_variety = str(env_config.get("crop_variety", "") or crop_variety).strip()

    scenario_bits = [f"crop={crop_name}"]
    if "year" in agro_params:
        scenario_bits.append(f"year={agro_params['year']}")
    if "latitude" in agro_params and "longitude" in agro_params:
        scenario_bits.append(
            f"lat={float(agro_params['latitude']):.2f}, lon={float(agro_params['longitude']):.2f}"
        )
    if env_config.get("scenario_id"):
        scenario_bits.append(f"scenario_id={env_config['scenario_id']}")

    return ", ".join(scenario_bits)


# ---------------------------------------------------------------------------
# Multiprocessing helpers (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _worker_validate(args, generator_cls, config, output_dir):
    """Validate a single env_config (weather completeness). Returns env_config or None."""
    env_config, idx, split = args
    worker_config = deepcopy(config)
    if worker_config.get("save_folder"):
        save_folder = Path(str(worker_config["save_folder"]))
        worker_config["save_folder"] = str(save_folder / f"worker_{os.getpid()}") + os.sep
    gen = generator_cls(worker_config, output_dir)
    try:
        gen._validate_weather(env_config)
        return idx, env_config
    except Exception:
        return idx, None


def _worker_render(args, generator_cls, config, output_dir):
    """Render prompt for a single env_config (no weather validation). Returns verl record or None."""
    env_config, idx, split = args
    gen = generator_cls(config, output_dir)
    try:
        return gen._convert_to_verl_format(env_config, idx, split)
    except Exception:
        return None


class WOFOSTDatasetGenerator(BaseDatasetGenerator):
    """Generate train/test splits for wofost_gym setups.

    This generator creates dataset configurations for WOFOST crop simulation
    environments. It supports VERL parquet format (for use with VERL's ToolAgentLoop).

    Supports a ``variants`` key in the config to generate multiple datasets
    (e.g. with/without crop traits, with/without thinking) from a single
    weather validation pass.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self.data_mode = self.config.get("data_mode", "single_crop")
        self.base_config = self._build_base_config()
        # Cross-split dedup: ensures no (crop, year, lat, lon) overlap between splits
        self._global_seen: set = set()

    def _build_base_config(self) -> Dict[str, Any]:
        base = {
            "env_id": self.config.get("env_id", "lnpkw-v0"),
            "agro_file": self.config.get("agro_file", "wheat_agro.yaml"),
            "wofost_gym_path": self.config.get("wofost_gym_path", DEFAULT_WOFOST_GYM_PATH),
            "llm_mode": self.config.get("llm_mode", True),
            "seed": self.config.get("seed", 0),
            "turn_num": self.config.get("turn_num", 241),
            "intvn_interval": self.config.get("intvn_interval", 1),
            "scale_action_amounts_by_interval": self.config.get("scale_action_amounts_by_interval", False),
            "require_think": self.config.get("require_think", False),
            "thinking_mode": self.config.get("thinking_mode", "grounding_decision"),
            "think_tag": self.config.get("think_tag", "tool_call"),
            "objective_id": self.config.get("objective_id", "profit_max"),
        }
        if self.config.get("prompt_action_schema_env_id"):
            base["prompt_action_schema_env_id"] = self.config["prompt_action_schema_env_id"]
        if self.config.get("save_folder"):
            base["save_folder"] = self.config["save_folder"]
        if self.config.get("weather_cache_dir"):
            base["weather_cache_dir"] = self.config["weather_cache_dir"]
        if "include_crop_traits" in self.config or self.config.get("include_crop_traits", False):
            base["include_crop_traits"] = self.config.get("include_crop_traits", False)
        if self.config.get("crop_traits_dir"):
            base["crop_traits_dir"] = self.config["crop_traits_dir"]
        if self.config.get("trait_schema"):
            base["trait_schema"] = self.config["trait_schema"]
        if self.config.get("prompt_objective_id"):
            base["prompt_objective_id"] = self.config["prompt_objective_id"]
        if self.config.get("prompt_objective_text"):
            base["prompt_objective_text"] = self.config["prompt_objective_text"]
        if self.config.get("reward_params"):
            base["reward_params"] = deepcopy(self.config["reward_params"])
        if self.config.get("include_profit_context"):
            base["include_profit_context"] = bool(self.config["include_profit_context"])
        if self.config.get("profit_context_params"):
            base["profit_context_params"] = deepcopy(self.config["profit_context_params"])
        if self.config.get("y_ref") is not None:
            base["y_ref"] = self.config["y_ref"]
        if self.config.get("fert_amount") is not None:
            base["fert_amount"] = self.config["fert_amount"]
        if self.config.get("irrig_amount") is not None:
            base["irrig_amount"] = self.config["irrig_amount"]
        # Only include non-empty optional fields
        if self.config.get("env_reward"):
            base["env_reward"] = self.config["env_reward"]
        if self.config.get("wofost_params"):
            base["wofost_params"] = deepcopy(self.config["wofost_params"])
        if self.config.get("agro_params"):
            base["agro_params"] = deepcopy(self.config["agro_params"])
        return base

    def _placeholder_prompt_env_label(self, env_config: Dict[str, Any]) -> str:
        return "WOFOST-Gym"

    def _placeholder_prompt_context(self, env_config: Dict[str, Any]) -> str:
        return _describe_placeholder_context(env_config)

    def splits(self) -> List[str]:
        splits = []
        if self.config.get("num_train_samples"):
            splits.append("train")
        if self.config.get("num_val_samples"):
            splits.append("val")
        if self.config.get("num_test_samples"):
            splits.append("test")
        if not splits:
            raise ValueError("Config must specify num_train_samples, num_val_samples, or num_test_samples.")
        return splits

    def build_split_records(self, split: str) -> List[Dict[str, Any]]:
        num_samples = self.config[f"num_{split}_samples"]
        if self.data_mode == "single_crop":
            return self._build_seed_records(split, num_samples)
        if self.data_mode == "multi_crop_ood":
            return self._build_seed_records(split, num_samples)
        raise ValueError(f"Unsupported data_mode: {self.data_mode}")

    def _get_split_crops(self, split: str) -> List[str]:
        if self.data_mode != "multi_crop_ood":
            return []
        if split == "train":
            key = "train_crops"
        else:
            # val and test both use eval_crops, falling back to test_crops
            key = "eval_crops" if "eval_crops" in self.config else "test_crops"
        crops = self.config.get(key, [])
        if not crops:
            raise ValueError(f"data_mode=multi_crop_ood requires non-empty '{key}'.")
        return [str(crop) for crop in crops]

    def _init_split_rng(self, split: str):
        """Initialize RNG and state for a split. Call once before _generate_more_records."""
        import numpy as np

        gen_seed = self.config.get("generation_seed", 42)
        num_train = self.config.get("num_train_samples", 0)
        num_val = self.config.get("num_val_samples", 0)
        margin = lambda n: n + n // 10  # 10% retry margin

        if split == "train":
            master_seed = gen_seed
            seed_offset = 0
        elif split == "val":
            master_seed = gen_seed + 1
            seed_offset = margin(num_train)
        else:  # test
            master_seed = gen_seed + 2
            seed_offset = margin(num_train) + margin(num_val)

        self._rng = np.random.RandomState(master_seed)
        self._seed_offset = seed_offset
        self._seen = set(self._global_seen)  # inherit all previously used keys
        self._generated = 0
        self._split = split
        self._split_crops = self._get_split_crops(split)

    def _generate_more_records(self, split: str, count: int) -> List[Dict[str, Any]]:
        """Generate `count` more unique records, continuing from current RNG state."""
        if split != getattr(self, "_split", None):
            self._init_split_rng(split)
        return self._generate_records_for_crops(count)

    def _get_crop_ranges(self, crop: str | None):
        """Return (lat_range, lon_range) for a crop from ``crop_ranges`` config.

        Falls back to global ``latitude_range`` / ``longitude_range`` only
        for single-crop mode (crop is None).
        """
        if crop is None:
            # Single-crop mode: use global ranges
            return (
                self.config.get("latitude_range", [40.0, 55.0]),
                self.config.get("longitude_range", [-10.0, 30.0]),
            )
        crop_ranges = self.config.get("crop_ranges", {})
        if crop not in crop_ranges:
            raise ValueError(
                f"No crop_ranges entry for '{crop}'. "
                f"Add crop_ranges.{crop}.latitude_range/longitude_range to your config."
            )
        per_crop = crop_ranges[crop]
        return per_crop["latitude_range"], per_crop["longitude_range"]

    def _generate_records_for_crops(
        self, count: int, fixed_crop: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate `count` records. If fixed_crop is set, all records use that crop."""
        year_range = self.config.get("year_range", [1984, 2019])

        records = []
        added = 0
        while added < count:
            if fixed_crop is not None:
                crop = fixed_crop
            elif self.data_mode == "multi_crop_ood":
                crop = self._split_crops[self._generated % len(self._split_crops)]
            else:
                crop = None

            lat_range, lon_range = self._get_crop_ranges(crop)
            year = int(self._rng.randint(year_range[0], year_range[1] + 1))
            lat = round(float(self._rng.uniform(lat_range[0], lat_range[1])), 2)
            lon = round(float(self._rng.uniform(lon_range[0], lon_range[1])), 2)

            key = (year, lat, lon)
            if crop is not None:
                key = (crop, year, lat, lon)
            if key in self._seen:
                continue
            self._seen.add(key)
            self._global_seen.add(key)

            cfg = deepcopy(self.base_config)
            split_metadata = _split_metadata(
                getattr(self, "_split", "unknown"),
                dataset_labels=self.config.get("labels"),
            )
            group_labels = {
                "simulator": self.config.get("env_name", "wofost_gym"),
                **split_metadata["group_labels"],
            }
            if cfg.get("objective_id"):
                group_labels["objective_id"] = str(cfg["objective_id"])
            if cfg.get("env_id"):
                group_labels["env_id"] = str(cfg["env_id"])
            if crop is not None:
                cfg["agro_file"] = f"{crop}_agro.yaml"
                cfg["crop_name"] = crop
                group_labels["crop"] = crop
            cfg["year"] = year
            cfg["seed"] = self._seed_offset + self._generated
            cfg["env_name"] = self.config.get("env_name", "wofost_gym")
            cfg["dataset_id"] = self.dataset_id
            cfg["dataset_split"] = getattr(self, "_split", "unknown")
            cfg["dataset_role"] = split_metadata["role"]
            if split_metadata.get("validation_set"):
                cfg["validation_set"] = split_metadata["validation_set"]
            cfg["scenario_id"] = scenario_id_for_row(
                cfg["dataset_split"],
                str(crop or _infer_crop_name(cfg)),
                year,
                lat,
                lon,
            )
            cfg["trajectory_group_labels"] = group_labels

            agro_params = deepcopy(cfg.get("agro_params", {}))
            agro_params["year"] = year
            agro_params["latitude"] = lat
            agro_params["longitude"] = lon
            cfg["agro_params"] = agro_params

            records.append(_sync_objective_group_label(cfg))
            added += 1
            self._generated += 1
        return records

    def _build_seed_records(self, split: str, num_samples: int) -> List[Dict[str, Any]]:
        """Seed-driven deterministic mapping to (year, lat, lon) combinations."""
        self._init_split_rng(split)
        return self._generate_more_records(split, num_samples)

    # ------------------------------------------------------------------
    # Weather validation (expensive) vs prompt rendering (cheap)
    # ------------------------------------------------------------------

    def _validate_weather(self, env_config: Dict[str, Any]) -> None:
        """Create env and run full no-op episode to verify weather and crop viability.

        Checks three things:
        1. Weather data is complete (no missing data errors).
        2. The crop reaches at least flowering (max DVS >= threshold).
           Filters unrealistic crop × climate combinations where the crop
           can never mature (e.g. millet at high latitudes).
        3. The crop produces non-zero yield (max WSO > 0).
           Filters cases where DVS advances but assimilation is zero due
           to extreme temperature mismatch (e.g. warm-adapted crops in
           cold climates: DVS ticks but no biomass grows).

        Raises on failure so the caller can retry with a different config.
        """
        min_dvs = self.config.get("min_dvs_threshold", 1.5)

        # Use a minimal config (no traits, no think) — only weather matters
        minimal = deepcopy(env_config)
        minimal["include_crop_traits"] = False
        minimal["require_think"] = False
        minimal["thinking_mode"] = "grounding_decision"
        env, _ = create_environment("wofost_gym", minimal)
        env.reset()
        max_dvs = 0.0
        max_wso = 0.0
        done = False
        while not done:
            _, _, done, info = env.step(0)
            metrics = info.get("turn_metrics", {})
            dvs = metrics.get("dvs", 0.0)
            wso = metrics.get("wso", 0.0)
            if dvs > max_dvs:
                max_dvs = dvs
            if wso > max_wso:
                max_wso = wso
        env.close()

        if max_dvs < min_dvs:
            raise ValueError(
                f"Crop did not reach min DVS threshold: max_dvs={max_dvs:.3f} < {min_dvs} "
                f"(crop={env_config.get('crop_name', '?')}, "
                f"lat={env_config.get('agro_params', {}).get('latitude', '?')}, "
                f"year={env_config.get('agro_params', {}).get('year', '?')})"
            )

        if max_wso <= 0:
            raise ValueError(
                f"Crop produced no yield: max_wso={max_wso:.1f} "
                f"(crop={env_config.get('crop_name', '?')}, "
                f"lat={env_config.get('agro_params', {}).get('latitude', '?')}, "
                f"year={env_config.get('agro_params', {}).get('year', '?')})"
            )

    def _render_prompt(self, env_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return the dataset placeholder prompt without touching the simulator."""
        return self._build_placeholder_prompt(env_config)

    def _build_initial_prompt(self, env_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build WOFOST-specific initial prompt for VERL (validate + render)."""
        self._validate_weather(env_config)
        return self._render_prompt(env_config)

    # ------------------------------------------------------------------
    # generate() — supports single-run and multi-variant modes
    # ------------------------------------------------------------------

    def generate(self) -> None:
        """Generate VERL-compatible parquet dataset.

        If ``variants`` is present in config, weather is validated once and
        prompts are rendered separately for each variant — much faster than
        running the full pipeline N times.

        Otherwise falls back to the standard single-dataset generation with
        per-crop balanced retry for multi_crop_ood mode.
        """
        variants = self.config.get("variants")
        if variants:
            return self._generate_with_variants(variants)
        if self.data_mode != "multi_crop_ood":
            return super().generate()
        return self._generate_multi_crop()

    def _generate_multi_crop(self) -> None:
        """Single-dataset generation with per-crop balanced retry."""
        import datasets
        from tqdm import tqdm

        save_dir = self.output_dir / self.dataset_id
        save_dir.mkdir(parents=True, exist_ok=True)
        num_workers = self.config.get("num_workers", 64)

        summary = {}
        for split in self.splits():
            target = self.config.get(f"num_{split}_samples")
            records = self.build_split_records(split)
            verl_records = []
            total_failed = 0

            while True:
                work_items = [(r, i, split) for i, r in enumerate(records)]
                desc = f"Generating {split}"
                if num_workers > 1:
                    worker_fn = partial(
                        _worker_convert,
                        generator_cls=type(self),
                        config=self.config,
                        output_dir=str(self.output_dir),
                    )
                    with mp.Pool(num_workers) as pool:
                        worker_results = list(tqdm(
                            pool.imap(worker_fn, work_items),
                            total=len(work_items), desc=desc,
                        ))
                        results = [res for _, res in worker_results]
                else:
                    results = [
                        self._convert_to_verl_format(r, i, split)
                        for i, r in tqdm(enumerate(records), total=len(records), desc=desc)
                    ]

                failed_crops: list[str] = []
                for rec, res in zip(records, results):
                    if res is None:
                        failed_crops.append(rec.get("crop_name", ""))
                    else:
                        verl_records.append(res)

                failed = len(failed_crops)
                total_failed += failed

                if failed == 0 or (target and len(verl_records) >= target):
                    break

                # Retry per-crop to maintain balance
                crop_deficit = Counter(failed_crops)
                records = []
                for crop, need in crop_deficit.items():
                    print(f"  {split}: {crop} has {need} failures, retrying...")
                    records.extend(self._generate_records_for_crops(need, fixed_crop=crop))

            if target:
                verl_records = verl_records[:target]
            if total_failed:
                print(f"  {split}: {total_failed} samples failed and replaced")
            summary[split] = len(verl_records)
            ds = datasets.Dataset.from_list(verl_records)
            parquet_path = save_dir / f"{split}.parquet"
            ds.to_parquet(str(parquet_path))

        self._write_manifest(save_dir, summary)
        self._print_summary(save_dir, summary)

    # ------------------------------------------------------------------
    # Multi-variant generation: validate once, render N times
    # ------------------------------------------------------------------

    def _generate_with_variants(self, variants: List[Dict[str, Any]]) -> None:
        """Validate weather once, then render prompts for each variant."""
        import datasets
        from tqdm import tqdm

        num_workers = self.config.get("num_workers", 64)

        # ---- Step 1: build & validate env configs per split ----
        validated: Dict[str, List[Dict[str, Any]]] = {}
        for split in self.splits():
            target = self.config.get(f"num_{split}_samples")
            records = self.build_split_records(split)
            good: List[Dict[str, Any]] = []
            total_failed = 0

            while True:
                work_items = [(r, i, split) for i, r in enumerate(records)]
                desc = f"Validating weather ({split})"
                if num_workers > 1:
                    worker_fn = partial(
                        _worker_validate,
                        generator_cls=type(self),
                        config=self.config,
                        output_dir=str(self.output_dir),
                    )
                    with mp.Pool(num_workers) as pool:
                        results = list(tqdm(
                            pool.imap(worker_fn, work_items),
                            total=len(work_items), desc=desc,
                        ))
                else:
                    results = []
                    for r in tqdm(records, desc=desc):
                        try:
                            self._validate_weather(r)
                            results.append(r)
                        except Exception:
                            results.append(None)

                failed_crops: list[str] = []
                for rec, res in zip(records, results):
                    if res is None:
                        failed_crops.append(rec.get("crop_name", ""))
                    else:
                        good.append(rec)

                failed = len(failed_crops)
                total_failed += failed

                if failed == 0 or (target and len(good) >= target):
                    break

                if self.data_mode == "multi_crop_ood":
                    crop_deficit = Counter(failed_crops)
                    records = []
                    for crop, need in crop_deficit.items():
                        print(f"  {split}: {crop} has {need} weather failures, retrying...")
                        records.extend(self._generate_records_for_crops(need, fixed_crop=crop))
                else:
                    need = target - len(good) if target else len(failed_crops)
                    print(f"  {split}: {failed} weather failures, retrying {need}...")
                    records = self._generate_more_records(split, need)

            if target:
                good = good[:target]
            if total_failed:
                print(f"  {split}: {total_failed} weather validations failed and replaced")
            validated[split] = good

        # ---- Step 2: render prompts for each variant ----
        for variant in variants:
            variant_id = variant["dataset_id"]
            # Prompt-only flags
            prompt_overrides = {
                k: v for k, v in variant.items() if k != "dataset_id"
            }

            save_dir = self.output_dir / variant_id
            save_dir.mkdir(parents=True, exist_ok=True)
            summary: Dict[str, int] = {}

            for split, configs in validated.items():
                desc = f"Rendering {variant_id}/{split}"
                verl_records = []

                def render_one(args):
                    cfg, idx = args
                    cfg_v = deepcopy(cfg)
                    cfg_v.update(prompt_overrides)
                    cfg_v = _sync_objective_group_label(cfg_v)
                    prompt = self._render_prompt(cfg_v)
                    return {
                        "data_source": f"{self.config.get('env_name', 'agrimanager')}/{variant_id}",
                        "agent_name": "agri_tool_agent",
                        "prompt": prompt,
                        "reward_model": {"style": "rule", "ground_truth": None},
                        "extra_info": {
                            "split": split,
                            "index": idx,
                            "interaction_kwargs": {
                                "name": "agri",
                                "env_config": cfg_v,
                            },
                        },
                    }

                items = list(enumerate(configs))
                if num_workers > 1:
                    with mp.Pool(num_workers) as pool:
                        verl_records = list(tqdm(
                            pool.imap(
                                partial(_worker_render_variant, prompt_overrides=prompt_overrides,
                                        generator_cls=type(self), config=self.config,
                                        output_dir=str(self.output_dir), variant_id=variant_id),
                                [(cfg, i, split) for i, cfg in enumerate(configs)],
                            ),
                            total=len(configs), desc=desc,
                        ))
                else:
                    verl_records = [
                        render_one((cfg, i))
                        for i, cfg in tqdm(enumerate(configs), total=len(configs), desc=desc)
                    ]

                # Filter out any render failures (should be rare)
                verl_records = [r for r in verl_records if r is not None]
                summary[split] = len(verl_records)
                ds = datasets.Dataset.from_list(verl_records)
                ds.to_parquet(str(save_dir / f"{split}.parquet"))

            variant_config = deepcopy(self.config)
            variant_config["dataset_id"] = variant_id
            original_config = self.config
            try:
                self.config = variant_config
                self._write_manifest(save_dir, summary)
            finally:
                self.config = original_config
            self._print_summary(save_dir, summary)


class WOFOSTArtifactDatasetBuilder(BaseDatasetGenerator):
    """Build immutable VERL-compatible dataset artifacts from a weather pool."""

    _PRIMARY_SPLIT_ORDER = ("train", "val", "test", "val_id", "val_drought", "val_wet", "val_hot", "val_cold")

    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        # Placeholder prompt rendering is cheap enough that single-process
        # execution is the sensible default, but callers can still override it.
        self.config.setdefault("num_workers", 1)
        self.source_cfg = deepcopy(self.config.get("source") or {})
        self.sampling_cfg = deepcopy(self.config.get("sampling") or {})
        self.env_cfg = deepcopy(self.config.get("env") or {})
        if self.config.get("variants"):
            raise ValueError("Artifact-first WOFOST datasets do not support config variants.")
        self._validate_config()
        self.pool_dir = self._resolve_pool_dir()
        self.y_ref_map = self._load_y_ref_map()
        self.base_env_config = self._materialize_base_env_config()
        self._scenario_set_records_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _validate_config(self) -> None:
        if self.source_cfg.get("kind") != "weather_pool":
            raise ValueError("WOFOST artifact builder requires source.kind=weather_pool.")
        if not self.source_cfg.get("path"):
            raise ValueError("WOFOST artifact builder requires source.path.")
        splits = self.sampling_cfg.get("splits")
        if not isinstance(splits, dict) or not splits:
            raise ValueError("WOFOST artifact builder requires sampling.splits.")
        scenario_sets = self.sampling_cfg.get("scenario_sets") or {}
        if scenario_sets and not isinstance(scenario_sets, dict):
            raise ValueError("sampling.scenario_sets must be a mapping when provided.")
        for set_name, set_cfg in scenario_sets.items():
            if not isinstance(set_cfg, dict):
                raise ValueError(f"sampling.scenario_sets.{set_name} must be a mapping.")
            self._validate_sampling_selector(
                set_cfg,
                path=f"sampling.scenario_sets.{set_name}",
            )
        for split_name, split_cfg in splits.items():
            if not isinstance(split_cfg, dict):
                raise ValueError(f"sampling.splits.{split_name} must be a mapping.")
            scenario_set = str(split_cfg.get("scenario_set") or "").strip()
            raw_slice = split_cfg.get("scenario_slice")
            if scenario_set:
                if scenario_set not in scenario_sets:
                    raise ValueError(
                        f"sampling.splits.{split_name}.scenario_set references unknown "
                        f"scenario set {scenario_set!r}."
                    )
                split_selector_keys = {
                    "crops",
                    "num_samples",
                    "crop_budgets",
                    "crop_variety_budgets",
                    "source_split",
                }
                present = sorted(key for key in split_selector_keys if key in split_cfg)
                if present:
                    raise ValueError(
                        f"sampling.splits.{split_name} uses scenario_set={scenario_set!r}, "
                        f"so scenario selection keys must live in sampling.scenario_sets."
                    )
                if raw_slice is not None:
                    _normalize_scenario_slice(raw_slice)
                continue
            if raw_slice is not None:
                raise ValueError(
                    f"sampling.splits.{split_name}.scenario_slice requires scenario_set."
                )
            self._validate_sampling_selector(
                split_cfg,
                path=f"sampling.splits.{split_name}",
            )

    def _validate_sampling_selector(self, cfg: Dict[str, Any], *, path: str) -> None:
        has_uniform = "crops" in cfg and "num_samples" in cfg
        has_budgets = "crop_budgets" in cfg
        has_variety_budgets = "crop_variety_budgets" in cfg
        if sum(bool(v) for v in (has_uniform, has_budgets, has_variety_budgets)) != 1:
            raise ValueError(
                f"{path} must define either 'crops + num_samples', "
                "'crop_budgets', or 'crop_variety_budgets'."
            )
        if ("crops" in cfg) != ("num_samples" in cfg):
            raise ValueError(f"{path} must define 'crops' and 'num_samples' together.")

    def _resolve_pool_dir(self) -> Path:
        local_cache_dir = self.source_cfg.get("local_cache_dir")
        if local_cache_dir is None:
            return ensure_pool(
                str(self.source_cfg["path"]),
                revision=str(self.source_cfg.get("revision", "main")),
            )
        return ensure_pool(
            str(self.source_cfg["path"]),
            revision=str(self.source_cfg.get("revision", "main")),
            local_dir=Path(local_cache_dir),
        )

    def _resolve_config_relative_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path))
        if path.is_absolute():
            return path
        config_path = self.config.get("_config_path")
        if config_path:
            return (Path(str(config_path)).resolve().parent / path).resolve()
        return (REPO_ROOT / path).resolve()

    def _load_y_ref_map(self) -> Dict[str, float]:
        raw_path = (
            self.config.get("y_ref_map_path")
            or self.config.get("calibrated_y_ref_path")
            or self.env_cfg.get("y_ref_map_path")
            or self.env_cfg.get("calibrated_y_ref_path")
        )
        if not raw_path:
            return {}

        path = self._resolve_config_relative_path(str(raw_path))
        if not path.is_file():
            require_file = bool(
                self.config.get(
                    "require_y_ref_map_file",
                    self.env_cfg.get("require_y_ref_map_file", True),
                )
            )
            if not require_file:
                return {}
            raise FileNotFoundError(
                f"Calibrated y_ref map not found: {path}. "
                "Run the T2.3 y_ref calibration helper before building this dataset."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "y_ref_by_scenario" in raw:
            raw = raw["y_ref_by_scenario"]
        if not isinstance(raw, dict):
            raise ValueError(f"Calibrated y_ref map must be a JSON object: {path}")

        y_ref_map: Dict[str, float] = {}
        for scenario_id, value in raw.items():
            try:
                y_ref = float(value)
            except (TypeError, ValueError):
                continue
            if y_ref > 0.0:
                y_ref_map[str(scenario_id)] = y_ref
        if not y_ref_map:
            raise ValueError(f"Calibrated y_ref map contains no positive values: {path}")
        return y_ref_map

    def _apply_calibrated_y_ref(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.y_ref_map:
            return env_config

        scenario_id = str(env_config.get("scenario_id", "") or "")
        y_ref = self.y_ref_map.get(scenario_id)
        if y_ref is None:
            require_match = bool(
                self.config.get(
                    "require_y_ref_map_match",
                    self.env_cfg.get("require_y_ref_map_match", True),
                )
            )
            if require_match and env_config.get("objective_id") in {
                "yield_max",
                "profit_max",
                "water_stewardship",
                "nutrient_stewardship",
            }:
                raise KeyError(
                    f"Scenario {scenario_id!r} is missing from calibrated y_ref map."
                )
            return env_config

        env_config["y_ref"] = float(y_ref)
        reward_params = deepcopy(env_config.get("reward_params") or {})
        reward_params["y_ref"] = float(y_ref)
        env_config["reward_params"] = reward_params
        return env_config

    def _materialize_base_env_config(self) -> Dict[str, Any]:
        """Resolve env defaults and path normalization via WOFOSTEnvConfig only."""
        env_cfg = deepcopy(self.env_cfg)
        bundled_cache_dir = find_pool_meteo_cache_dir(self.pool_dir)
        if bundled_cache_dir is not None and "weather_cache_dir" not in env_cfg:
            env_cfg["weather_cache_dir"] = str(bundled_cache_dir)
        return _drop_implicit_save_folder(
            WOFOSTEnvConfig(**env_cfg).to_dict(),
            env_cfg,
        )

    def _materialize_split_env_config(self, split_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve base env config plus split-local env overrides."""
        split_env_cfg = split_cfg.get("env") or {}
        if not isinstance(split_env_cfg, dict):
            raise ValueError("sampling.splits.<split>.env must be a mapping when provided.")
        if not split_env_cfg:
            return deepcopy(self.base_env_config)

        merged_env_cfg = deepcopy(self.env_cfg)
        merged_env_cfg.update(deepcopy(split_env_cfg))
        bundled_cache_dir = find_pool_meteo_cache_dir(self.pool_dir)
        if bundled_cache_dir is not None and "weather_cache_dir" not in merged_env_cfg:
            merged_env_cfg["weather_cache_dir"] = str(bundled_cache_dir)
        return _drop_implicit_save_folder(
            WOFOSTEnvConfig(**merged_env_cfg).to_dict(),
            merged_env_cfg,
        )

    def _split_sampling_cfg(self, split: str) -> Dict[str, Any]:
        split_cfg = self.sampling_cfg["splits"][split]
        scenario_set = str(split_cfg.get("scenario_set") or "").strip()
        if not scenario_set:
            return split_cfg

        scenario_sets = self.sampling_cfg.get("scenario_sets") or {}
        scenario_cfg = deepcopy(scenario_sets[scenario_set])
        if "seed" in split_cfg:
            scenario_cfg["seed"] = split_cfg["seed"]
        return scenario_cfg

    def _split_seed(self, split: str) -> int:
        sampling_cfg = self._split_sampling_cfg(split)
        return int(sampling_cfg.get("seed", self.sampling_cfg.get("generation_seed", 42)))

    def _scenario_set_cache_key(self, split: str) -> str:
        split_cfg = self.sampling_cfg["splits"][split]
        scenario_set = str(split_cfg.get("scenario_set") or "").strip()
        scenario_cfg = self._split_sampling_cfg(split)
        payload = json.dumps(
            {"scenario_set": scenario_set, "scenario_cfg": scenario_cfg},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _apply_scenario_slice(
        self,
        scenarios: List[Dict[str, Any]],
        split_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        scenario_slice = _normalize_scenario_slice(split_cfg.get("scenario_slice"))
        if scenario_slice is None:
            return list(scenarios)
        subset = scenarios[
            scenario_slice["start"]:scenario_slice["stop"]:scenario_slice["step"]
        ]
        if not subset:
            raise ValueError(
                "scenario_slice selected zero scenarios for "
                f"split {split_cfg!r}."
            )
        return list(subset)

    def splits(self) -> List[str]:
        return self._split_order()

    def _split_order(self) -> List[str]:
        split_cfg = self.sampling_cfg.get("splits") or {}
        ordered = []
        for split in self._PRIMARY_SPLIT_ORDER:
            if split in split_cfg:
                ordered.append(split)
        for split in split_cfg:
            if split not in ordered:
                ordered.append(split)
        return ordered

    def split_metadata(self, split: str) -> Dict[str, Any]:
        split_cfg = (self.sampling_cfg.get("splits") or {}).get(split) or {}
        return _split_metadata(
            split,
            split_cfg=split_cfg,
            dataset_labels=self.config.get("labels"),
        )

    def _placeholder_prompt_env_label(self, env_config: Dict[str, Any]) -> str:
        return "WOFOST-Gym"

    def _placeholder_prompt_context(self, env_config: Dict[str, Any]) -> str:
        return _describe_placeholder_context(env_config)

    def _sample_by_crop_budgets(
        self,
        pool: Dict[str, Any],
        crop_budgets: Dict[str, int],
        seed: int,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        scenarios_by_crop: Dict[str, List[Dict[str, Any]]] = {}
        for crop, raw_target in crop_budgets.items():
            target = int(raw_target)
            if crop not in pool:
                raise ValueError(
                    f"Crop '{crop}' not found in weather pool. "
                    f"Available crops: {sorted(pool.keys())}"
                )
            df = pool[crop]
            rng = np.random.RandomState(crop_sampling_seed(seed, crop))
            indices = rng.permutation(len(df))
            crop_rows: List[Dict[str, Any]] = []
            seen: set[tuple[str, int, float, float]] = set()
            for idx in indices:
                if len(crop_rows) >= target:
                    break
                row = df.iloc[int(idx)]
                year = int(row["year"])
                lat = round(float(row["latitude"]), 2)
                lon = round(float(row["longitude"]), 2)
                key = (crop, year, lat, lon)
                if key in seen:
                    continue
                seen.add(key)
                crop_rows.append(
                    {
                        "crop_name": crop,
                        "year": year,
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
            if len(crop_rows) != target:
                raise ValueError(
                    f"Not enough unique weather-pool rows for crop '{crop}': "
                    f"need {target}, got {len(crop_rows)}."
                )
            scenarios_by_crop[crop] = crop_rows

        scenarios: List[Dict[str, Any]] = []
        crop_order = list(crop_budgets.keys())
        max_rounds = max(len(rows) for rows in scenarios_by_crop.values())
        for round_index in range(max_rounds):
            for crop in crop_order:
                rows = scenarios_by_crop[crop]
                if round_index < len(rows):
                    scenarios.append(rows[round_index])
        return scenarios

    def _variety_split_label(self, crop: str, crop_variety: str) -> str:
        variety_splits = self.sampling_cfg.get("variety_splits") or {}
        crop_splits = variety_splits.get(crop) if isinstance(variety_splits, dict) else None
        if not isinstance(crop_splits, dict):
            return "unknown"
        for split_label, varieties in crop_splits.items():
            if str(crop_variety) in {str(v) for v in varieties or []}:
                return str(split_label)
        return "unknown"

    def _sample_rows_for_crop(
        self,
        pool: Dict[str, Any],
        crop: str,
        target: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        if crop not in pool:
            raise ValueError(
                f"Crop '{crop}' not found in weather pool. "
                f"Available crops: {sorted(pool.keys())}"
            )
        df = pool[crop]
        rng = np.random.RandomState(crop_sampling_seed(seed, crop))
        indices = rng.permutation(len(df))
        crop_rows: List[Dict[str, Any]] = []
        seen: set[tuple[str, int, float, float]] = set()
        for idx in indices:
            if len(crop_rows) >= target:
                break
            row = df.iloc[int(idx)]
            year = int(row["year"])
            lat = round(float(row["latitude"]), 2)
            lon = round(float(row["longitude"]), 2)
            key = (crop, year, lat, lon)
            if key in seen:
                continue
            seen.add(key)
            crop_rows.append(
                {
                    "crop_name": crop,
                    "year": year,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        if len(crop_rows) != target:
            raise ValueError(
                f"Not enough unique weather-pool rows for crop '{crop}': "
                f"need {target}, got {len(crop_rows)}."
            )
        return crop_rows

    def _sample_by_crop_variety_budgets(
        self,
        pool: Dict[str, Any],
        crop_variety_budgets: Dict[str, Dict[str, int]],
        seed: int,
    ) -> List[Dict[str, Any]]:
        scenarios: List[Dict[str, Any]] = []
        for crop, raw_variety_budgets in crop_variety_budgets.items():
            if not isinstance(raw_variety_budgets, dict) or not raw_variety_budgets:
                raise ValueError(
                    "crop_variety_budgets entries must map crop names to "
                    "non-empty {variety: count} mappings."
                )
            variety_budgets = {
                str(variety): int(target)
                for variety, target in raw_variety_budgets.items()
            }
            max_target = max(variety_budgets.values())
            crop_rows = self._sample_rows_for_crop(pool, str(crop), max_target, seed)
            for round_index, base_row in enumerate(crop_rows):
                for crop_variety, target in variety_budgets.items():
                    if round_index >= target:
                        continue
                    row = dict(base_row)
                    row["crop_variety"] = crop_variety
                    row["variety_split"] = self._variety_split_label(str(crop), crop_variety)
                    scenarios.append(row)
        return scenarios

    def _resolve_split_scenarios(self, split: str) -> List[Dict[str, Any]]:
        split_cfg = self.sampling_cfg["splits"][split]
        scenario_set = str(split_cfg.get("scenario_set") or "").strip()
        if scenario_set:
            cache_key = self._scenario_set_cache_key(split)
            cached = self._scenario_set_records_cache.get(cache_key)
            if cached is None:
                scenario_cfg = self._split_sampling_cfg(split)
                source_split = str(scenario_cfg.get("source_split") or "").strip()
                if not source_split:
                    role = self.split_metadata(split)["role"]
                    source_split = {
                        "train": "train",
                        "validation": "val",
                        "test": "test",
                    }.get(role, split)
                split_dir = self.pool_dir / source_split
                if not split_dir.is_dir():
                    raise FileNotFoundError(f"Weather pool split directory not found: {split_dir}")
                pool = load_pool(split_dir)
                seed = self._split_seed(split)
                if "crop_variety_budgets" in scenario_cfg:
                    cached = self._sample_by_crop_variety_budgets(
                        pool,
                        scenario_cfg["crop_variety_budgets"],
                        seed,
                    )
                elif "crop_budgets" in scenario_cfg:
                    cached = self._sample_by_crop_budgets(pool, scenario_cfg["crop_budgets"], seed)
                else:
                    cached = sample_scenarios(
                        pool=pool,
                        crops=list(scenario_cfg["crops"]),
                        num_samples=int(scenario_cfg["num_samples"]),
                        seed=seed,
                    )
                self._scenario_set_records_cache[cache_key] = list(cached)
            return self._apply_scenario_slice(cached, split_cfg)

        scenario_cfg = self._split_sampling_cfg(split)
        source_split = str(scenario_cfg.get("source_split") or "").strip()
        if not source_split:
            role = self.split_metadata(split)["role"]
            source_split = {
                "train": "train",
                "validation": "val",
                "test": "test",
            }.get(role, split)
        split_dir = self.pool_dir / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Weather pool split directory not found: {split_dir}")
        pool = load_pool(split_dir)
        seed = self._split_seed(split)
        if "crop_variety_budgets" in scenario_cfg:
            return self._sample_by_crop_variety_budgets(
                pool,
                scenario_cfg["crop_variety_budgets"],
                seed,
            )
        if "crop_budgets" in scenario_cfg:
            return self._sample_by_crop_budgets(pool, scenario_cfg["crop_budgets"], seed)
        return sample_scenarios(
            pool=pool,
            crops=list(scenario_cfg["crops"]),
            num_samples=int(scenario_cfg["num_samples"]),
            seed=seed,
        )

    def build_split_records(self, split: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        split_cfg = self.sampling_cfg["splits"][split]
        split_metadata = self.split_metadata(split)
        split_base_env_config = self._materialize_split_env_config(split_cfg)
        for scenario in self._resolve_split_scenarios(split):
            crop_name = str(scenario["crop_name"])
            crop_variety = str(scenario.get("crop_variety", "") or "").strip()
            variety_split = str(scenario.get("variety_split", "") or "").strip()
            year = int(scenario["year"])
            latitude = round(float(scenario["latitude"]), 2)
            longitude = round(float(scenario["longitude"]), 2)
            if crop_variety:
                scenario_id = scenario_id_for_variety_row(
                    split,
                    crop_name,
                    crop_variety,
                    year,
                    latitude,
                    longitude,
                )
            else:
                scenario_id = scenario_id_for_row(
                    split,
                    crop_name,
                    year,
                    latitude,
                    longitude,
                )
            paired_scenario_id = paired_scenario_id_for_row(
                crop_name,
                year,
                latitude,
                longitude,
                crop_variety=crop_variety or None,
            )
            group_labels = {
                "simulator": "wofost_gym",
                "crop": crop_name,
                **split_metadata["group_labels"],
            }
            if split_base_env_config.get("objective_id"):
                group_labels["objective_id"] = str(split_base_env_config["objective_id"])
            if split_base_env_config.get("env_id"):
                group_labels["env_id"] = str(split_base_env_config["env_id"])
            if crop_variety:
                group_labels["variety"] = crop_variety
            if variety_split:
                group_labels["variety_split"] = variety_split

            scenario_overrides = {
                "crop_name": crop_name,
                "agro_file": f"{crop_name}_agro.yaml",
                "year": year,
                "seed": env_seed_from_scenario_id(scenario_id),
                "scenario_id": scenario_id,
                "paired_scenario_id": paired_scenario_id,
                "dataset_id": self.dataset_id,
                "dataset_split": split,
                "dataset_role": split_metadata["role"],
                "trajectory_group_labels": group_labels,
            }
            if split_metadata.get("validation_set"):
                scenario_overrides["validation_set"] = split_metadata["validation_set"]
            if crop_variety:
                scenario_overrides["crop_variety"] = crop_variety
            if variety_split:
                scenario_overrides["variety_split"] = variety_split
            env_config = deepcopy(split_base_env_config)
            env_config.update(scenario_overrides)
            env_config = self._apply_calibrated_y_ref(env_config)

            agro_params = deepcopy(env_config.get("agro_params") or {})
            agro_params["year"] = year
            agro_params["latitude"] = latitude
            agro_params["longitude"] = longitude
            if crop_variety:
                agro_params["crop_variety"] = crop_variety
            env_config["agro_params"] = agro_params
            env_config = _drop_implicit_save_folder(
                WOFOSTEnvConfig(**env_config).to_dict(),
                {**self.env_cfg, **(split_cfg.get("env") or {})},
            )
            env_config["env_name"] = self.config.get("env_name", "wofost_gym")
            env_config["trajectory_group_labels"] = group_labels
            env_config = apply_split_metadata_to_env_config(
                env_config,
                split,
                split_cfg,
                dataset_labels=self.config.get("labels"),
                inferred_labels={"weather_regime": _WEATHER_REGIME_SPLIT_LABELS.get(split, "")},
            )
            env_config = _sync_objective_group_label(env_config)
            if not env_config.get("wofost_params"):
                env_config.pop("wofost_params", None)
            records.append(env_config)
        return records

    def _convert_to_verl_format(
        self, env_config: Dict[str, Any], idx: int, split: str
    ) -> Dict[str, Any]:
        env_config = _sync_objective_group_label(dict(env_config))
        return {
            "data_source": f"{self.config.get('env_name', 'wofost_gym')}/{self.dataset_id}",
            "agent_name": "agri_tool_agent",
            "prompt": self._build_initial_prompt(env_config),
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {
                "split": split,
                "index": idx,
                "scenario_id": env_config["scenario_id"],
                "interaction_kwargs": {
                    "name": "agri",
                    "env_config": env_config,
                },
            },
        }

    def _generate_more_records(self, split: str, count: int) -> List[Dict[str, Any]]:
        raise RuntimeError(
            "WOFOST artifact datasets do not support runtime record replacement. "
            "Fix the dataset config or source weather pool instead."
        )

    def generate(self) -> None:
        super().generate()


def _worker_render_variant(args, prompt_overrides, generator_cls, config, output_dir, variant_id):
    """Render a single record for a variant (multiprocessing helper)."""
    cfg, idx, split = args
    gen = generator_cls(config, output_dir)
    try:
        cfg_v = deepcopy(cfg)
        cfg_v.update(prompt_overrides)
        cfg_v = _sync_objective_group_label(cfg_v)
        prompt = gen._render_prompt(cfg_v)
        return {
            "data_source": f"{config.get('env_name', 'agrimanager')}/{variant_id}",
            "agent_name": "agri_tool_agent",
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {
                "split": split,
                "index": idx,
                "interaction_kwargs": {
                    "name": "agri",
                    "env_config": cfg_v,
                },
            },
        }
    except Exception:
        return None


def generate(config: Dict[str, Any], output_dir: str):
    """Entry point used by dataset build entrypoints."""
    if (
        "source" in config
        and "sampling" in config
        and "env" in config
    ):
        builder = WOFOSTArtifactDatasetBuilder(config, output_dir)
    else:
        builder = WOFOSTDatasetGenerator(config, output_dir)
    builder.generate()
