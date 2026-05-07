"""Dataset generation for cycles_gym environments."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from typing import Any, Dict, List

from agrimanager.env.base.objective_prompt import DEFAULT_OBJECTIVE_ID
from agrimanager.env.base import BaseDatasetGenerator


def _location_from_env_id(env_id: str) -> str | None:
    for location in ("RockSprings", "NewHolland"):
        if location in env_id:
            return location
    return None


def _task_from_env_id(env_id: str) -> str:
    if env_id.startswith("CropPlanning"):
        return "crop_planning"
    if env_id.startswith("Corn"):
        return "corn"
    return "unknown"


def _compact_env_id_label(env_id: str) -> str:
    parts: list[str] = []
    if env_id.startswith("CropPlanning"):
        parts.append("plan")
    elif env_id.startswith("Corn"):
        parts.append("corn")
        if "Short" in env_id:
            parts.append("short")
    else:
        parts.append("env")

    location = _location_from_env_id(env_id)
    if location == "RockSprings":
        parts.append("rs")
    elif location == "NewHolland":
        parts.append("nh")

    if "RW" in env_id:
        parts.append("rw")
    elif "FW" in env_id:
        parts.append("fw")

    return "_".join(parts)


def _year_window_from_env_config(env_config: Dict[str, Any]) -> str | None:
    env_kwargs = env_config.get("env_kwargs") or {}
    start_year = env_kwargs.get("start_year")
    end_year = env_kwargs.get("end_year")
    if start_year is None or end_year is None:
        return None
    return f"{start_year}-{end_year}"


def _default_group_labels(env_config: Dict[str, Any], split: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}

    raw_labels = env_config.get("trajectory_group_labels", {})
    if isinstance(raw_labels, dict):
        for key, value in raw_labels.items():
            key_str = str(key or "").strip()
            value_str = str(value or "").strip()
            if key_str and value_str:
                labels[key_str] = value_str

    env_id = str(env_config.get("env_id") or "").strip()
    if env_id:
        labels.setdefault("env_id", _compact_env_id_label(env_id))
        labels.setdefault("task", _task_from_env_id(env_id))
        location = _location_from_env_id(env_id)
        if location:
            labels.setdefault("location", location)

    labels.setdefault("dataset_split", split)

    year_window = _year_window_from_env_config(env_config)
    if year_window:
        labels.setdefault("year_window", year_window)

    return labels


class CyclesDatasetGenerator(BaseDatasetGenerator):
    """Generate parquet datasets for cycles_gym experiments."""

    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self.base_config = deepcopy(self.config.get("base_config", {}))
        for key in (
            "objective_id",
            "prompt_objective_id",
            "prompt_objective_text",
            "reward_params",
            "include_profit_context",
            "profit_context_params",
        ):
            if key in self.config and key not in self.base_config:
                self.base_config[key] = deepcopy(self.config[key])
        self.base_config.setdefault("objective_id", DEFAULT_OBJECTIVE_ID)

    def splits(self) -> List[str]:
        splits = []
        if self.config.get("train_configs") or self.config.get("train_generation"):
            splits.append("train")
        if self.config.get("val_configs") or self.config.get("val_generation"):
            splits.append("val")
        if self.config.get("test_configs") or self.config.get("test_generation"):
            splits.append("test")
        if not splits:
            raise ValueError(
                "Config must specify at least one generated or explicit split definition."
            )
        return splits

    def build_split_records(self, split: str) -> List[Dict[str, Any]]:
        entries = self.config.get(f"{split}_configs")
        generated = self.config.get(f"{split}_generation")
        if entries and generated:
            raise ValueError(
                f"Specify only one of '{split}_configs' or '{split}_generation', not both."
            )
        if generated:
            return self._expand_generated_records(split, generated)
        if not entries:
            raise ValueError(
                f"Config must specify '{split}_configs' as a non-empty list or '{split}_generation'."
            )

        records = []
        for idx, entry in enumerate(entries):
            cfg = deepcopy(self.base_config)
            cfg.update(deepcopy(entry))
            records.append(self._finalize_record(cfg, split=split, row_index=idx))
        return records

    def _expand_generated_records(self, split: str, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        env_ids = list(spec.get("env_ids") or [self.base_config.get("env_id")])
        year_windows = list(spec.get("year_windows") or [])
        seeds_per_combo = int(spec.get("seeds_per_combo", 0))
        seed_start = int(spec.get("seed_start", 0))
        paired_price_regimes = spec.get("paired_price_regimes")

        if not env_ids:
            raise ValueError(f"'{split}_generation.env_ids' must be non-empty.")
        if not year_windows:
            raise ValueError(f"'{split}_generation.year_windows' must be non-empty.")
        if seeds_per_combo <= 0:
            raise ValueError(f"'{split}_generation.seeds_per_combo' must be > 0.")
        if paired_price_regimes:
            return self._expand_paired_price_regime_records(
                split=split,
                env_ids=env_ids,
                year_windows=year_windows,
                seeds_per_combo=seeds_per_combo,
                seed_start=seed_start,
                paired_price_regimes=paired_price_regimes,
            )

        records: List[Dict[str, Any]] = []
        row_index = 0
        combo_index = 0
        for env_id in env_ids:
            for raw_window in year_windows:
                start_year, end_year = self._normalize_year_window(raw_window)
                for seed_offset in range(seeds_per_combo):
                    cfg = deepcopy(self.base_config)
                    cfg["env_id"] = env_id
                    cfg["seed"] = seed_start + combo_index * seeds_per_combo + seed_offset
                    env_kwargs = deepcopy(cfg.get("env_kwargs") or {})
                    env_kwargs["start_year"] = start_year
                    env_kwargs["end_year"] = end_year
                    cfg["env_kwargs"] = env_kwargs
                    records.append(self._finalize_record(cfg, split=split, row_index=row_index))
                    row_index += 1
                combo_index += 1
        return records

    def _normalize_year_window(self, raw_window: Any) -> tuple[int, int]:
        if isinstance(raw_window, dict):
            start_year = raw_window.get("start_year")
            end_year = raw_window.get("end_year")
        elif isinstance(raw_window, (list, tuple)) and len(raw_window) == 2:
            start_year, end_year = raw_window
        else:
            raise ValueError(
                "Year windows must be two-item lists/tuples or dicts with start_year/end_year."
            )

        if start_year is None or end_year is None:
            raise ValueError("Year window is missing start_year or end_year.")

        start_year = int(start_year)
        end_year = int(end_year)
        if end_year < start_year:
            raise ValueError(f"Invalid year window: {start_year}>{end_year}")
        return start_year, end_year

    def _drop_empty_dicts(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                value = self._drop_empty_dicts(value)
                if isinstance(value, dict) and not value:
                    continue
                cleaned[key] = value
            return cleaned
        if isinstance(obj, list):
            return [self._drop_empty_dicts(value) for value in obj]
        return obj

    def _stable_seed(self, split: str, row_index: int, cfg: Dict[str, Any]) -> int:
        payload = {
            "dataset_id": self.dataset_id,
            "split": split,
            "row_index": row_index,
            "env_id": cfg.get("env_id"),
            "env_kwargs": cfg.get("env_kwargs", {}),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return int(digest[:8], 16)

    def _sample_crop_prices(
        self,
        cfg: Dict[str, Any],
        *,
        split: str,
        row_index: int,
    ) -> Dict[str, Any]:
        env_kwargs = deepcopy(cfg.get("env_kwargs") or {})
        explicit_crop_prices = env_kwargs.get("crop_prices")
        sampling_spec = deepcopy(env_kwargs.pop("crop_price_sampling", None))

        if explicit_crop_prices is not None and sampling_spec is not None:
            raise ValueError("Specify only one of env_kwargs.crop_prices or env_kwargs.crop_price_sampling.")

        if not sampling_spec:
            cfg["env_kwargs"] = env_kwargs
            return cfg

        if not isinstance(sampling_spec, dict):
            raise ValueError("env_kwargs.crop_price_sampling must be a dict keyed by crop name.")
        sampled_prices = self._sample_prices_from_spec(
            sampling_spec,
            seed_payload={
                "dataset_id": self.dataset_id,
                "split": split,
                "row_index": row_index,
                "env_id": cfg.get("env_id"),
                "seed": cfg.get("seed"),
                "env_kwargs": env_kwargs,
                "crop_price_sampling": sampling_spec,
            },
        )

        env_kwargs["crop_prices"] = sampled_prices
        cfg["env_kwargs"] = env_kwargs
        return cfg


    def _sample_prices_from_spec(
        self,
        sampling_spec: Dict[str, Any],
        *,
        seed_payload: Dict[str, Any],
    ) -> Dict[str, float]:
        if not isinstance(sampling_spec, dict):
            raise ValueError("crop-price sampling spec must be a dict keyed by crop name.")

        digest = hashlib.sha1(
            json.dumps(seed_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        sampled_prices: Dict[str, float] = {}
        for crop_name, crop_spec in sampling_spec.items():
            if not isinstance(crop_spec, dict):
                raise ValueError(
                    f"crop-price sampling spec for {crop_name!r} must be a dict with min/max[/mode]."
                )

            min_price = crop_spec.get("min")
            max_price = crop_spec.get("max")
            if min_price is None or max_price is None:
                raise ValueError(
                    f"crop-price sampling spec for {crop_name!r} must define both 'min' and 'max'."
                )

            min_price = float(min_price)
            max_price = float(max_price)
            mode_price = crop_spec.get("mode")
            if mode_price is None:
                mode_price = (min_price + max_price) / 2.0
            mode_price = float(mode_price)

            if min_price > max_price:
                raise ValueError(
                    f"crop-price sampling spec for {crop_name!r} has min > max: {min_price}>{max_price}"
                )
            if not (min_price <= mode_price <= max_price):
                raise ValueError(
                    f"crop-price sampling spec for {crop_name!r} must satisfy min <= mode <= max."
                )

            round_ndigits = int(crop_spec.get("round_ndigits", 2))
            sampled_prices[str(crop_name)] = round(
                rng.triangular(min_price, max_price, mode_price),
                round_ndigits,
            )

        return sampled_prices


    def _paired_scenario_id(self, cfg: Dict[str, Any]) -> str:
        env_kwargs = deepcopy(cfg.get("env_kwargs") or {})
        env_kwargs.pop("crop_prices", None)
        env_kwargs.pop("crop_price_sampling", None)
        payload = {
            "env_id": cfg.get("env_id"),
            "seed": cfg.get("seed"),
            "env_kwargs": env_kwargs,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"cycles_pair_{digest[:16]}"


    def _expand_paired_price_regime_records(
        self,
        *,
        split: str,
        env_ids: List[str],
        year_windows: List[Any],
        seeds_per_combo: int,
        seed_start: int,
        paired_price_regimes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        base_price_sampling = deepcopy(
            paired_price_regimes.get("base_price_sampling")
            or (self.base_config.get("env_kwargs") or {}).get("crop_price_sampling")
        )
        regimes = deepcopy(paired_price_regimes.get("regimes") or {})
        if not base_price_sampling:
            raise ValueError("paired_price_regimes must define base_price_sampling or base_config.env_kwargs.crop_price_sampling.")
        if not regimes:
            raise ValueError("paired_price_regimes.regimes must be a non-empty mapping.")

        records: List[Dict[str, Any]] = []
        row_index = 0
        combo_index = 0
        for env_id in env_ids:
            for raw_window in year_windows:
                start_year, end_year = self._normalize_year_window(raw_window)
                for seed_offset in range(seeds_per_combo):
                    cfg = deepcopy(self.base_config)
                    cfg["env_id"] = env_id
                    cfg["seed"] = seed_start + combo_index * seeds_per_combo + seed_offset
                    env_kwargs = deepcopy(cfg.get("env_kwargs") or {})
                    env_kwargs["start_year"] = start_year
                    env_kwargs["end_year"] = end_year
                    env_kwargs.pop("crop_prices", None)
                    env_kwargs.pop("crop_price_sampling", None)
                    cfg["env_kwargs"] = env_kwargs

                    pair_id = self._paired_scenario_id(cfg)
                    base_payload = {
                        "dataset_id": self.dataset_id,
                        "split": split,
                        "paired_scenario_id": pair_id,
                        "price_regime": "id_base",
                        "seed": cfg.get("seed"),
                    }
                    base_prices = self._sample_prices_from_spec(
                        base_price_sampling,
                        seed_payload=base_payload,
                    )

                    for regime_name, regime_spec in regimes.items():
                        regime_cfg = deepcopy(cfg)
                        regime_env_kwargs = deepcopy(regime_cfg.get("env_kwargs") or {})
                        regime_prices = deepcopy(base_prices)

                        if not isinstance(regime_spec, dict):
                            raise ValueError(
                                f"paired_price_regimes.regimes[{regime_name!r}] must be a dict."
                            )

                        fixed_prices = regime_spec.get("crop_prices") or {}
                        fixed_overrides = regime_spec.get("crop_price_overrides") or {}
                        sampled_overrides = regime_spec.get("crop_price_sampling_overrides") or {}
                        multipliers = regime_spec.get("crop_price_multipliers") or {}

                        for crop_name, value in fixed_prices.items():
                            regime_prices[str(crop_name)] = float(value)
                        for crop_name, value in fixed_overrides.items():
                            regime_prices[str(crop_name)] = float(value)
                        for crop_name, value in multipliers.items():
                            base_value = regime_prices.get(str(crop_name))
                            if base_value is None:
                                raise ValueError(
                                    f"paired_price_regimes.regimes[{regime_name!r}].crop_price_multipliers references missing crop {crop_name!r}."
                                )
                            regime_prices[str(crop_name)] = round(
                                float(base_value) * float(value),
                                2,
                            )
                        if sampled_overrides:
                            sampled_prices = self._sample_prices_from_spec(
                                sampled_overrides,
                                seed_payload={
                                    "dataset_id": self.dataset_id,
                                    "split": split,
                                    "paired_scenario_id": pair_id,
                                    "price_regime": regime_name,
                                    "seed": regime_cfg.get("seed"),
                                },
                            )
                            regime_prices.update(sampled_prices)

                        regime_env_kwargs["crop_prices"] = regime_prices
                        regime_cfg["env_kwargs"] = regime_env_kwargs
                        regime_cfg["price_regime"] = str(regime_name)
                        regime_cfg["paired_scenario_id"] = pair_id
                        regime_cfg["paired_base_crop_prices"] = deepcopy(base_prices)
                        regime_cfg["price_regime_metadata"] = deepcopy(
                            regime_spec.get("metadata") or {}
                        )

                        records.append(
                            self._finalize_record(regime_cfg, split=split, row_index=row_index)
                        )
                        row_index += 1
                combo_index += 1

        return records

    def _scenario_id(self, split: str, cfg: Dict[str, Any]) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "dataset_split": split,
            "env_id": cfg.get("env_id"),
            "seed": cfg.get("seed"),
            "env_kwargs": cfg.get("env_kwargs", {}),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"cycles_{digest[:16]}"

    def _placeholder_prompt_context(self, env_config: Dict[str, Any]) -> str | None:
        env_id = str(env_config.get("env_id", "") or "").strip()
        env_kwargs = env_config.get("env_kwargs") or {}
        start_year = env_kwargs.get("start_year")
        end_year = env_kwargs.get("end_year")
        if start_year is not None and end_year is not None:
            return f"{env_id}, {start_year}-{end_year}"
        return env_id or None

    def _finalize_record(
        self,
        cfg: Dict[str, Any],
        *,
        split: str,
        row_index: int,
    ) -> Dict[str, Any]:
        cfg = self._sample_crop_prices(cfg, split=split, row_index=row_index)
        cfg = self._drop_empty_dicts(cfg)
        env_name = self.config.get("env_name", "cycles_gym")
        cfg.setdefault("env_name", env_name)
        cfg["dataset_id"] = self.dataset_id
        cfg["dataset_split"] = split
        cfg["seed"] = int(cfg.get("seed")) if cfg.get("seed") is not None else self._stable_seed(split, row_index, cfg)
        cfg["scenario_id"] = self._scenario_id(split, cfg)
        group_labels = _default_group_labels(cfg, split)
        group_labels.setdefault("simulator", env_name)
        group_labels.setdefault("split", split)
        group_labels.setdefault("objective_id", str(cfg.get("objective_id", "profit_max")))
        price_regime = cfg.get("price_regime")
        if price_regime:
            group_labels.setdefault("price_regime", str(price_regime))
        price_regime_metadata = cfg.get("price_regime_metadata") or {}
        regime_family = price_regime_metadata.get("regime_family")
        if regime_family:
            group_labels.setdefault("regime_family", str(regime_family))
        shocked_crop = price_regime_metadata.get("shocked_crop")
        if shocked_crop:
            group_labels.setdefault("shocked_crop", str(shocked_crop))
        if str(cfg.get("env_id", "")).startswith("Corn"):
            group_labels.setdefault("crop", "maize")
            cfg.setdefault("crop_name", "maize")
        cfg["trajectory_group_labels"] = group_labels
        return cfg


def generate(config: Dict[str, Any], output_dir: str) -> None:
    """Entry point used by entrypoints/dataset/build.sh."""
    builder = CyclesDatasetGenerator(config, output_dir)
    builder.generate()
