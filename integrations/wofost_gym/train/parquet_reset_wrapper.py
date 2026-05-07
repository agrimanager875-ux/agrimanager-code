"""Gym wrapper that injects deterministic reset configs on each reset.

The wrapper name is legacy. Primary WOFOST training now passes reset configs
from parquet dataset artifacts.
"""

import gymnasium as gym
import numpy as np
from typing import Callable


class ParquetResetWrapper(gym.Wrapper):
    """Intercepts reset() to inject parquet-defined reset settings.

    Placed between wrap_env_reward() and RecordEpisodeStatistics in the
    wrapper chain so that normalization wrappers still work correctly.

    Args:
        env: The wrapped gym environment.
        configs: Either legacy list[(year, (lat, lon))] or
            list[{"year": int, "location": (lat, lon), "seed": int,
            "agro_file": str|None}].
        rng: numpy random Generator for reproducible sampling.
        sample_with_replacement: If False, iterate configs in shuffled cycles.
        make_env_for_agro_file: Optional factory that rebuilds the wrapped
            environment for a specific agro_file. Required for dynamic
            agro_file switching.
        current_agro_file: Agro file currently loaded by ``env`` at init.
    """

    def __init__(
        self,
        env: gym.Env,
        configs: list[dict | tuple[int, tuple[float, float]]],
        rng=None,
        sample_with_replacement: bool = False,
        make_env_for_agro_file: Callable[[str], gym.Env] | None = None,
        current_agro_file: str | None = None,
    ):
        super().__init__(env)
        if not configs:
            raise ValueError("configs must not be empty")
        self.configs = [self._normalize_config(cfg) for cfg in configs]
        self.rng = rng or np.random.default_rng()
        self.sample_with_replacement = sample_with_replacement
        self.make_env_for_agro_file = make_env_for_agro_file
        self._current_agro_file = current_agro_file
        self._order = np.arange(len(self.configs), dtype=np.int64)
        self._cursor = 0
        if not self.sample_with_replacement:
            self.rng.shuffle(self._order)

    @staticmethod
    def _normalize_config(cfg: dict | tuple[int, tuple[float, float]]) -> dict:
        if isinstance(cfg, dict):
            if "year" not in cfg or "location" not in cfg:
                raise ValueError(f"Invalid config dict, missing year/location: {cfg}")
            year = int(cfg["year"])
            location = cfg["location"]
            agro_file = cfg.get("agro_file")
            seed = cfg.get("seed")
        elif isinstance(cfg, tuple) and len(cfg) == 2:
            year, location = cfg
            year = int(year)
            agro_file = None
            seed = None
        else:
            raise ValueError(f"Unsupported config format: {cfg}")

        if not isinstance(location, (list, tuple)) or len(location) != 2:
            raise ValueError(f"Invalid location in config: {cfg}")

        normalized = {
            "year": year,
            "location": (float(location[0]), float(location[1])),
            "seed": None if seed is None else int(seed),
            "agro_file": None if agro_file in (None, "") else str(agro_file),
        }
        return normalized

    def _next_index(self) -> int:
        if self.sample_with_replacement:
            return int(self.rng.integers(len(self.configs)))

        if self._cursor >= len(self._order):
            self.rng.shuffle(self._order)
            self._cursor = 0

        idx = int(self._order[self._cursor])
        self._cursor += 1
        return idx

    def _switch_env_if_needed(self, agro_file: str | None):
        if agro_file is None or agro_file == self._current_agro_file:
            return

        if self.make_env_for_agro_file is None:
            raise RuntimeError(
                "Parquet config requires agro_file switching but "
                "make_env_for_agro_file is not provided."
            )

        new_env = self.make_env_for_agro_file(agro_file)
        old_env = self.env
        self.env = new_env
        self._current_agro_file = agro_file

        try:
            old_env.close()
        except Exception:
            pass

        print(f"[ParquetResetWrapper] switched agro_file to {agro_file}")

    def reset(self, **kwargs):
        last_error = None
        max_attempts = len(self.configs)

        for _ in range(max_attempts):
            idx = self._next_index()
            cfg = self.configs[idx]
            year = int(cfg["year"])
            location = cfg["location"]
            seed = cfg.get("seed")
            agro_file = cfg["agro_file"]

            # Build fresh kwargs per attempt so previous setdefault values
            # (year/location) do not leak across retries.
            attempt_kwargs = dict(kwargs)
            attempt_kwargs.setdefault("year", year)
            attempt_kwargs.setdefault("location", location)
            if seed is not None:
                attempt_kwargs.setdefault("seed", seed)

            try:
                self._switch_env_if_needed(agro_file)
                return self.env.reset(**attempt_kwargs)
            except Exception as exc:
                last_error = exc
                msg = str(exc)
                is_weather_error = ("WeatherDataProviderError" in type(exc).__name__) or ("No weather data" in msg)
                if is_weather_error:
                    print(
                        f"[ParquetResetWrapper] skip invalid weather config: "
                        f"agro_file={agro_file}, year={year}, location={location}, error={exc}"
                    )
                    continue
                raise

        raise RuntimeError(
            "All parquet reset configurations failed. "
            "Please re-generate/clean the parquet dataset or verify weather data source."
        ) from last_error
