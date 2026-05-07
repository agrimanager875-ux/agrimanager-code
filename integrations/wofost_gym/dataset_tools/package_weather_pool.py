#!/usr/bin/env python3
"""Bundle the required NASA POWER cache files into a WOFOST weather pool."""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CACHE_DIR = (
    SCRIPT_DIR.parents[3]
    / "AgriManagerExternal"
    / "WOFOSTGym"
    / "pcse"
    / "pcse"
    / ".pcse"
    / "meteo_cache"
)
DEFAULT_ARCHIVE_NAME = "meteo_cache.tar.gz"
_WOFOST_GYM_ROOT = SCRIPT_DIR.parents[3] / "AgriManagerExternal" / "WOFOSTGym"
_PCSE_IMPORT_ROOT = _WOFOST_GYM_ROOT / "pcse"
_CACHE_NAME_RE = re.compile(r"NASAPowerWeatherDataProvider_LAT(-?\d+)_LON(-?\d+)\.cache$")


def _iter_pool_parquets(pool_dir: Path):
    split_dirs = sorted(
        path
        for path in pool_dir.iterdir()
        if path.is_dir() and path.name != "meteo_cache"
    )
    if not split_dirs:
        raise FileNotFoundError(f"No split directories found under: {pool_dir}")
    for split_dir in split_dirs:
        yield from sorted(split_dir.glob("*.parquet"))


def _required_cache_years(pool_dir: Path, year_padding: int) -> dict[str, set[int]]:
    cache_years: dict[str, set[int]] = {}
    for parquet_path in _iter_pool_parquets(pool_dir):
        df = pd.read_parquet(parquet_path, columns=["year", "latitude", "longitude"])
        for year, lat, lon in zip(df["year"], df["latitude"], df["longitude"]):
            name = (
                "NASAPowerWeatherDataProvider_"
                f"LAT{int(float(lat) * 10):05d}_"
                f"LON{int(float(lon) * 10):05d}.cache"
            )
            years = cache_years.setdefault(name, set())
            target_year = int(year)
            for offset in range(-year_padding, year_padding + 1):
                years.add(target_year + offset)
    return cache_years


def _trim_cache_file(src: Path, dst: Path, years: set[int]) -> int:
    with open(src, "rb") as f:
        store, elevation, longitude, latitude, description, et_model = pickle.load(f)

    filtered_store = {
        key: value
        for key, value in store.items()
        if isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], date)
        and key[0].year in years
    }
    if not filtered_store:
        raise ValueError(f"No weather rows remain after trimming {src.name} to years {sorted(years)}")

    with open(dst, "wb") as f:
        payload = (filtered_store, elevation, longitude, latitude, description, et_model)
        pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)
    return dst.stat().st_size


def _parse_cache_name(name: str) -> tuple[float, float]:
    match = _CACHE_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized NASA cache filename: {name}")
    lat_code, lon_code = match.groups()
    return int(lat_code) / 10.0, int(lon_code) / 10.0


def _retrieve_source_cache(source_cache_dir: Path, name: str) -> Path:
    lat, lon = _parse_cache_name(name)
    source_cache_dir.mkdir(parents=True, exist_ok=True)

    if str(_PCSE_IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PCSE_IMPORT_ROOT))

    previous_cache_dir = os.environ.get("PCSE_METEO_CACHE_DIR")
    os.environ["PCSE_METEO_CACHE_DIR"] = str(source_cache_dir)
    try:
        from pcse.nasapower import NASAPowerWeatherDataProvider

        print(f"Retrieving missing/corrupt NASA weather cache for lat={lat:.1f}, lon={lon:.1f}")
        NASAPowerWeatherDataProvider(latitude=lat, longitude=lon, force_update=True)
    finally:
        if previous_cache_dir is None:
            os.environ.pop("PCSE_METEO_CACHE_DIR", None)
        else:
            os.environ["PCSE_METEO_CACHE_DIR"] = previous_cache_dir

    repaired_path = source_cache_dir / name
    if not repaired_path.is_file() or repaired_path.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to retrieve a valid NASA cache file for {name}")
    return repaired_path


def _package_one_cache(
    source_cache_dir: Path,
    dest_dir: Path,
    name: str,
    years: set[int],
    retrieve_missing: bool,
) -> tuple[str, str, int]:
    src = source_cache_dir / name
    dst = dest_dir / name
    if not src.exists():
        if not retrieve_missing:
            return name, "missing", 0
        src = _retrieve_source_cache(source_cache_dir, name)
    if dst.exists():
        return name, "skipped", 0
    try:
        size = _trim_cache_file(src, dst, years)
    except (EOFError, pickle.UnpicklingError):
        if not retrieve_missing:
            return name, "missing", 0
        src = _retrieve_source_cache(source_cache_dir, name)
        size = _trim_cache_file(src, dst, years)
    return name, "copied", size


def build_meteo_cache_archive(
    pool_dir: Path,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
) -> Path:
    dest_dir = pool_dir / "meteo_cache"
    if not dest_dir.is_dir():
        raise FileNotFoundError(f"Bundled weather cache directory not found: {dest_dir}")

    archive_path = pool_dir / archive_name
    tmp_archive_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_archive_path.exists():
        tmp_archive_path.unlink()

    print(f"Creating weather-cache archive: {archive_path}")
    with tarfile.open(tmp_archive_path, "w:gz") as tar:
        tar.add(dest_dir, arcname="meteo_cache")
    tmp_archive_path.replace(archive_path)
    return archive_path


def package_weather_pool(
    pool_dir: Path,
    source_cache_dir: Path,
    year_padding: int = 1,
    num_workers: int = 16,
    clean: bool = False,
    create_archive: bool = True,
    archive_only: bool = False,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    retrieve_missing: bool = True,
) -> None:
    pool_dir = pool_dir.resolve()
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Weather pool directory not found: {pool_dir}")

    dest_dir = pool_dir / "meteo_cache"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = pool_dir / archive_name

    copied = 0
    skipped = 0
    total_bytes = 0
    missing: list[str] = []
    required: list[str] = []

    if not archive_only:
        source_cache_dir = source_cache_dir.resolve()
        if not source_cache_dir.is_dir():
            raise FileNotFoundError(f"Source meteo cache directory not found: {source_cache_dir}")

        required_years = _required_cache_years(pool_dir, year_padding)
        required = sorted(required_years)

        if clean:
            required_set = set(required)
            for stale_path in dest_dir.glob("*.cache"):
                if stale_path.name not in required_set:
                    stale_path.unlink()
            if archive_path.exists():
                archive_path.unlink()

        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
            futures = {
                executor.submit(
                    _package_one_cache,
                    source_cache_dir,
                    dest_dir,
                    name,
                    required_years[name],
                    retrieve_missing,
                ): name
                for name in required
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                name, status, size = future.result()
                if status == "missing":
                    missing.append(name)
                elif status == "skipped":
                    skipped += 1
                else:
                    copied += 1
                    total_bytes += size
                if idx % 1000 == 0 or idx == len(required):
                    print(f"[{idx}/{len(required)}] packaged cache files")

        print(
            {
                "pool_dir": str(pool_dir),
                "dest_dir": str(dest_dir),
                "required_cache_files": len(required),
                "year_padding": year_padding,
                "num_workers": num_workers,
                "copied": copied,
                "skipped": skipped,
                "missing": len(missing),
                "retrieve_missing": bool(retrieve_missing),
                "size_gb": round(total_bytes / 1024 / 1024 / 1024, 3),
            }
        )
        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"Missing {len(missing)} required weather cache files under {source_cache_dir}. "
                f"Examples: {preview}"
            )

    archive_size_gb = None
    if create_archive:
        archive_path = build_meteo_cache_archive(pool_dir, archive_name=archive_name)
        archive_size_gb = round(archive_path.stat().st_size / 1024 / 1024 / 1024, 3)

    print(
        {
            "pool_dir": str(pool_dir),
            "dest_dir": str(dest_dir),
            "archive_path": str(archive_path) if create_archive else None,
            "archive_size_gb": archive_size_gb,
            "archive_only": archive_only,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the required NASA POWER cache files into a weather-pool directory.",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        required=True,
        help="Weather pool directory containing one or more split directories with parquet files.",
    )
    parser.add_argument(
        "--source-cache-dir",
        type=Path,
        default=DEFAULT_SOURCE_CACHE_DIR,
        help="Source PCSE meteo_cache directory to copy from.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete stale cache files from <pool-dir>/meteo_cache before copying.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip creating meteo_cache.tar.gz after packaging.",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Only create meteo_cache.tar.gz from an existing <pool-dir>/meteo_cache directory.",
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help=f"Archive filename to create under <pool-dir> (default: {DEFAULT_ARCHIVE_NAME}).",
    )
    parser.add_argument(
        "--year-padding",
        type=int,
        default=1,
        help="Include this many neighboring years on both sides of each scenario year.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers used for trimming and writing cache files.",
    )
    parser.add_argument(
        "--no-retrieve",
        action="store_false",
        dest="retrieve_missing",
        help="Fail on missing or corrupt source cache files instead of retrieving from NASA POWER.",
    )
    parser.set_defaults(retrieve_missing=True)
    args = parser.parse_args()
    package_weather_pool(
        args.pool_dir,
        args.source_cache_dir,
        year_padding=args.year_padding,
        num_workers=args.num_workers,
        clean=args.clean,
        create_archive=not args.no_archive,
        archive_only=args.archive_only,
        archive_name=args.archive_name,
        retrieve_missing=args.retrieve_missing,
    )


if __name__ == "__main__":
    main()
