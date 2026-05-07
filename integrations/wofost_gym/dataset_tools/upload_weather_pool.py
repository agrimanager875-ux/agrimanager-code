#!/usr/bin/env python3
"""Upload a prepared WOFOST weather pool directory to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "agrimanager/wofost-weather-pool"
DEFAULT_ARCHIVE_NAME = "meteo_cache.tar.gz"
REMOTE_DIRS = ("backups", "meteo_cache")
REMOTE_FILES = (DEFAULT_ARCHIVE_NAME,)


def _split_dirs(pool_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in pool_dir.iterdir()
        if path.is_dir()
        and path.name not in {"meteo_cache", "backups"}
        and any(path.glob("*.parquet"))
    )


def upload_weather_pool(
    pool_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    num_workers: int = 16,
    clear_remote: bool = True,
) -> None:
    pool_dir = pool_dir.resolve()
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Weather pool directory not found: {pool_dir}")
    split_dirs = _split_dirs(pool_dir)
    if not split_dirs:
        raise FileNotFoundError(f"No split parquet directories found under: {pool_dir}")
    archive_path = pool_dir / DEFAULT_ARCHIVE_NAME
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing weather-cache archive: {archive_path}. "
            "Run package_weather_pool.py first."
        )

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    if clear_remote:
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        remote_dirs = set(REMOTE_DIRS) | {path.name for path in split_dirs}
        for remote_dir in sorted(remote_dirs):
            has_remote_dir = any(path == remote_dir or path.startswith(f"{remote_dir}/") for path in remote_files)
            if has_remote_dir:
                print(f"Deleting remote folder: {remote_dir}/")
                api.delete_folder(
                    path_in_repo=remote_dir,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Remove stale {remote_dir} before weather-pool refresh",
                )
        for remote_file in REMOTE_FILES:
            if remote_file in remote_files:
                print(f"Deleting remote file: {remote_file}")
                api.delete_file(
                    path_in_repo=remote_file,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Remove stale {remote_file} before weather-pool refresh",
                )

    print(f"Uploading weather pool from {pool_dir} to hf://datasets/{repo_id}")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=pool_dir,
        allow_patterns=[
            "*/*.parquet",
            DEFAULT_ARCHIVE_NAME,
            "README.md",
            "*.csv",
            "*.json",
        ],
        num_workers=max(1, int(num_workers)),
        print_report=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a WOFOST weather pool directory to Hugging Face.",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        required=True,
        help="Local weather pool directory containing split parquet directories and meteo_cache.tar.gz.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Parallel workers used by huggingface_hub.upload_large_folder().",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="Do not delete existing train/val/test/meteo_cache/backups/archive before uploading.",
    )
    args = parser.parse_args()
    upload_weather_pool(
        args.pool_dir,
        repo_id=args.repo_id,
        num_workers=args.num_workers,
        clear_remote=not args.keep_remote,
    )


if __name__ == "__main__":
    main()
