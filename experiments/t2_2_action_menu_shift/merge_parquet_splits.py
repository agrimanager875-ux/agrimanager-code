#!/usr/bin/env python
"""Merge per-menu WOFOST parquet splits for action-menu-shift training."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    return parser.parse_args()


def merge_split(paths: list[str], output_path: Path) -> None:
    parts = [
        datasets.Dataset.from_parquet(str(Path(path).resolve()))
        for path in paths
    ]
    merged = datasets.concatenate_datasets(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(output_path))


def main() -> None:
    args = parse_args()
    save_dir = Path(args.output_dir) / args.dataset_id
    merge_split(args.train, save_dir / "train.parquet")
    merge_split(args.val, save_dir / "val.parquet")
    print(f"Merged action-menu dataset written to {save_dir}")


if __name__ == "__main__":
    main()
