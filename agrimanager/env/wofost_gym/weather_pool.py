"""Weather pool management for WOFOST dataset generation.

A weather pool is a collection of pre-validated (crop, year, latitude, longitude)
entries stored as per-crop parquet files. Using a pool eliminates the expensive
weather validation step; generation becomes a simple seed-based sample.

Pool structure (local or HuggingFace repo)::

    pool_dir/
      train/
        wheat.parquet     # columns: year, latitude, longitude
        maize.parquet
        ...
      val/
        wheat.parquet
        millet.parquet
        ...
      test/                 # optional; some benchmark pools are validation-only
        wheat.parquet
        millet.parquet
        ...
      val_drought/          # optional custom benchmark split
        chickpea.parquet
        potato.parquet

Usage::

    pool_dir = ensure_pool("agrimanager/wofost-weather-pool")
    pool = load_pool(pool_dir / "train")
    scenarios = sample_scenarios(pool, crops=["wheat", "maize"], num_samples=100, seed=42)
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


# Default download location: repo-local hidden cache directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POOL_DIR = _REPO_ROOT / ".cache" / "wofost_gym" / "weather_pool_20crop_3200_val128_test512"
DEFAULT_POOL_REPO_ID = "agrimanager/wofost-weather-pool"
METEO_CACHE_ARCHIVE_NAME = "meteo_cache.tar.gz"


# ---------------------------------------------------------------------------
# Download / locate pool
# ---------------------------------------------------------------------------

def download_pool(
    repo_id: str,
    local_dir: str | Path = DEFAULT_POOL_DIR,
    revision: str = "main",
) -> Path:
    """Download a weather pool from HuggingFace Hub.

    Downloads to ``local_dir`` and returns the directory containing split
    subdirectories with per-crop parquet files.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    if local_dir.exists() and _looks_like_pool_dir(local_dir):
        if _should_refresh_legacy_default_pool(repo_id, local_dir):
            print(
                f"Weather pool at {local_dir} has no test/ split; "
                "refreshing from HuggingFace..."
            )
        else:
            print(f"Weather pool already exists at {local_dir}")
            _ensure_pool_meteo_cache(local_dir)
            return local_dir

    print(f"Downloading weather pool from HuggingFace: {repo_id} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        revision=revision,
        allow_patterns=[
            "*.parquet",
            "*/*.parquet",
            METEO_CACHE_ARCHIVE_NAME,
            "meteo_cache/*.cache",
        ],
        ignore_patterns=["backups/**"],
    )
    if not _looks_like_pool_dir(local_dir):
        raise FileNotFoundError(
            f"Downloaded weather pool to {local_dir}, but no parquet files were "
            "found at the root or under split subdirectories."
        )
    _ensure_pool_meteo_cache(local_dir)
    print(f"Weather pool downloaded to {local_dir}")
    return local_dir


def ensure_pool(
    pool_path: str,
    revision: str = "main",
    local_dir: str | Path | None = None,
) -> Path:
    """Ensure weather pool is available locally.

    * If *pool_path* is a local directory with parquet files, use it directly.
    * Otherwise treat it as a HuggingFace repo ID and download to a repo-local
      cache under ``.cache/wofost_gym/``.

    Returns:
        Path to the local pool directory.
    """
    local = Path(pool_path)
    if local.is_dir() and _looks_like_pool_dir(local):
        return local

    if local.exists():
        raise FileNotFoundError(
            f"Weather pool not found at '{pool_path}'. "
            "Expected parquet files at the pool root or under split "
            "subdirectories such as train/ and val/."
        )

    # Treat namespace/name strings as HuggingFace repo IDs.
    if _looks_like_hf_repo_id(pool_path):
        cached_dir = Path(local_dir) if local_dir is not None else _default_local_dir_for_repo(pool_path)
        if cached_dir.is_dir() and _looks_like_pool_dir(cached_dir):
            if _should_refresh_legacy_default_pool(pool_path, cached_dir):
                return download_pool(pool_path, local_dir=cached_dir, revision=revision)
            _ensure_pool_meteo_cache(cached_dir)
            return cached_dir
        return download_pool(pool_path, local_dir=cached_dir, revision=revision)

    raise FileNotFoundError(
        f"Weather pool not found at '{pool_path}'. "
        f"Provide a local directory with per-crop parquet files, "
        f"or a HuggingFace repo ID (e.g. 'agrimanager/wofost-weather-pool')."
    )


def find_pool_meteo_cache_dir(pool_dir: str | Path) -> Path | None:
    """Return bundled meteo_cache directory when present."""
    return _ensure_pool_meteo_cache(Path(pool_dir))


def _looks_like_pool_dir(pool_dir: Path) -> bool:
    """Return True for both legacy flat pools and split train/val/test pools."""
    if any(pool_dir.glob("*.parquet")):
        return True
    return any(pool_dir.glob("*/*.parquet"))


def _pool_meteo_archive_path(pool_dir: Path) -> Path:
    return pool_dir / METEO_CACHE_ARCHIVE_NAME


def _has_pool_meteo_bundle(pool_dir: Path) -> bool:
    return (pool_dir / "meteo_cache").is_dir() or _pool_meteo_archive_path(pool_dir).is_file()


def _ensure_pool_meteo_cache(pool_dir: Path) -> Path | None:
    cache_dir = pool_dir / "meteo_cache"
    if cache_dir.is_dir():
        return cache_dir.resolve()

    archive_path = _pool_meteo_archive_path(pool_dir)
    if not archive_path.is_file():
        return None

    print(f"Extracting bundled weather cache from {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target_path = (pool_dir / member.name).resolve()
            if pool_dir.resolve() not in {target_path, *target_path.parents}:
                raise ValueError(f"Refusing to extract unsafe archive member: {member.name}")
        tar.extractall(path=pool_dir, filter="data")

    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Expected extracted meteo_cache directory at {cache_dir} after unpacking {archive_path}."
        )
    return cache_dir.resolve()


def _looks_like_legacy_split_pool(pool_dir: Path) -> bool:
    """Return True for old active pools that had train/val but no test split."""
    return (
        (pool_dir / "train").is_dir()
        and (pool_dir / "val").is_dir()
        and not (pool_dir / "test").is_dir()
    )


def _should_refresh_legacy_default_pool(repo_id: str, pool_dir: Path) -> bool:
    return repo_id == DEFAULT_POOL_REPO_ID and _looks_like_legacy_split_pool(pool_dir)


def _looks_like_hf_repo_id(pool_path: str) -> bool:
    """Heuristic for HuggingFace repo IDs without confusing local paths."""
    if pool_path.startswith(("/", ".", "~")) or "\\" in pool_path:
        return False
    parts = PurePosixPath(pool_path).parts
    return (
        len(parts) == 2
        and all(parts)
        and all(part not in {".", ".."} for part in parts)
    )


def _default_local_dir_for_repo(repo_id: str) -> Path:
    if repo_id == DEFAULT_POOL_REPO_ID:
        return DEFAULT_POOL_DIR
    slug = repo_id.replace("/", "__")
    return _REPO_ROOT / ".cache" / "wofost_gym" / slug


# ---------------------------------------------------------------------------
# Load pool
# ---------------------------------------------------------------------------

def load_pool(pool_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load weather pool from a local directory.

    Returns:
        Dict mapping crop name → DataFrame with columns
        ``year`` (int), ``latitude`` (float), ``longitude`` (float).
    """
    pool_dir = Path(pool_dir)
    pool: Dict[str, pd.DataFrame] = {}
    for pf in sorted(pool_dir.glob("*.parquet")):
        crop_name = pf.stem
        df = pd.read_parquet(pf)
        for col in ("year", "latitude", "longitude"):
            if col not in df.columns:
                raise ValueError(f"{pf}: missing required column '{col}'")
        pool[crop_name] = df
    if not pool:
        raise FileNotFoundError(f"No parquet files found in {pool_dir}")
    return pool


# ---------------------------------------------------------------------------
# Sample scenarios
# ---------------------------------------------------------------------------

def sample_scenarios(
    pool: Dict[str, pd.DataFrame],
    crops: List[str],
    num_samples: int,
    seed: int,
    exclude: Optional[Set[tuple]] = None,
) -> List[Dict[str, Any]]:
    """Sample validated scenarios from the weather pool.

    Draws equally from each crop (round-robin), ensuring no duplicates
    and no overlap with *exclude*.

    Args:
        pool: Dict mapping crop name → DataFrame.
        crops: Crop names to sample from.
        num_samples: Total samples to draw.
        seed: Random seed for reproducibility.
        exclude: Set of ``(crop, year, lat, lon)`` tuples to skip.

    Returns:
        List of scenario dicts, each with keys:
        ``crop_name``, ``year``, ``latitude``, ``longitude``.
    """
    rng = np.random.RandomState(seed)
    exclude = exclude or set()

    # Round-robin allocation
    per_crop = num_samples // len(crops)
    remainder = num_samples % len(crops)

    scenarios_by_crop: Dict[str, List[Dict[str, Any]]] = {}
    used: Set[tuple] = set(exclude)

    for i, crop in enumerate(crops):
        if crop not in pool:
            raise ValueError(
                f"Crop '{crop}' not found in weather pool. "
                f"Available crops: {sorted(pool.keys())}"
            )
        target = per_crop + (1 if i < remainder else 0)
        df = pool[crop]

        # Shuffle indices
        indices = rng.permutation(len(df))

        count = 0
        crop_scenarios: List[Dict[str, Any]] = []
        for idx in indices:
            if count >= target:
                break
            row = df.iloc[int(idx)]
            year = int(row["year"])
            lat = round(float(row["latitude"]), 2)
            lon = round(float(row["longitude"]), 2)
            key = (crop, year, lat, lon)
            if key in used:
                continue
            used.add(key)
            crop_scenarios.append({
                "crop_name": crop,
                "year": year,
                "latitude": lat,
                "longitude": lon,
            })
            count += 1

        if count < target:
            raise ValueError(
                f"Not enough unique scenarios for '{crop}': "
                f"need {target}, pool has {len(df)} entries, "
                f"got {count} after dedup."
            )
        scenarios_by_crop[crop] = crop_scenarios

    scenarios: List[Dict[str, Any]] = []
    for round_index in range(max(len(entries) for entries in scenarios_by_crop.values())):
        for crop in crops:
            entries = scenarios_by_crop[crop]
            if round_index < len(entries):
                scenarios.append(entries[round_index])

    return scenarios
