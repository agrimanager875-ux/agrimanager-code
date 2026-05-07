#!/usr/bin/env python3
"""Build a weather pool by generating and validating WOFOST scenarios.

Consumes weather-pool generation configs from ``weather_pool_configs/`` and
reuses the legacy scenario-generation + DVS-filtering pipeline. It outputs
per-crop parquet files (a weather pool), not final VERL-format datasets.

Usage:
    # Build one crop shard of the 20-crop reusable weather pool
    python integrations/wofost_gym/dataset_tools/build_weather_pool.py \
        pool_crop_wheat \
        --output-dir integrations/wofost_gym/dataset_tools/weather_pool_20crop_3200_val128_test512 \
        --num-workers 32

    # Smoke test with the small weather-pool config
    python integrations/wofost_gym/dataset_tools/build_weather_pool.py \
        pool_4id_2ood_test \
        --output-dir /tmp/pool_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
AGRIMANAGER_ROOT = SCRIPT_DIR.parents[2]
CONFIG_DIR = SCRIPT_DIR / "weather_pool_configs"

if str(AGRIMANAGER_ROOT) not in sys.path:
    sys.path.insert(0, str(AGRIMANAGER_ROOT))

from agrimanager.env.wofost_gym.create_dataset import WOFOSTDatasetGenerator


def build_pool(
    config_path: Path,
    output_dir: Path,
    num_workers: int | None = None,
    weather_cache_dir: Path | None = None,
) -> None:
    """Generate and validate weather scenarios, then save as per-crop pool.

    Reuses WOFOSTDatasetGenerator's full pipeline:
    1. Random (crop, year, lat, lon) generation with per-split seeding
    2. Weather validation (DVS >= threshold, WSO > 0)
    3. Per-crop balanced retry on failures
    4. Cross-split deduplication

    The only difference from ``generate_dataset.sh``: instead of rendering
    prompts and writing VERL parquets, we extract the validated scenario
    fields and write per-crop pool parquets.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if num_workers is not None:
        config["num_workers"] = num_workers
    if weather_cache_dir is not None:
        config["weather_cache_dir"] = str(weather_cache_dir.resolve())

    # Pool configs don't need dataset_id, but WOFOSTDatasetGenerator requires
    # it (inherited from BaseDatasetGenerator). Provide a dummy value.
    config.setdefault("dataset_id", "_pool_build")

    gen = WOFOSTDatasetGenerator(config, str(output_dir))

    # ── Step 1: generate + validate (reuses _generate_with_variants step 1) ──
    import multiprocessing as mp
    from collections import Counter
    from functools import partial

    from tqdm import tqdm

    from agrimanager.env.wofost_gym.create_dataset import _worker_validate

    workers = config.get("num_workers", 64)
    validated: dict[str, list[dict]] = {}

    for split in gen.splits():
        target = config.get(f"num_{split}_samples")
        records = gen.build_split_records(split)
        good: list[dict] = []
        total_failed = 0

        while True:
            work_items = [(r, i, split) for i, r in enumerate(records)]
            desc = f"Validating weather ({split})"

            if workers > 1:
                worker_fn = partial(
                    _worker_validate,
                    generator_cls=type(gen),
                    config=config,
                    output_dir=str(output_dir),
                )
                with mp.Pool(workers) as pool:
                    results = [None] * len(work_items)
                    for idx, res in tqdm(
                        pool.imap_unordered(worker_fn, work_items),
                        total=len(work_items),
                        desc=desc,
                    ):
                        results[idx] = res
            else:
                results = []
                for r in tqdm(records, desc=desc):
                    try:
                        gen._validate_weather(r)
                        results.append(r)
                    except Exception:
                        results.append(None)

            failed_crops: list[str] = []
            for rec, res in zip(records, results):
                if res is None:
                    failed_crops.append(rec.get("crop_name", ""))
                else:
                    good.append(res)

            failed = len(failed_crops)
            total_failed += failed

            if failed == 0 or (target and len(good) >= target):
                break

            if gen.data_mode == "multi_crop_ood":
                crop_deficit = Counter(failed_crops)
                records = []
                for crop, need in crop_deficit.items():
                    print(f"  {split}: {crop} has {need} weather failures, retrying...")
                    records.extend(gen._generate_records_for_crops(need, fixed_crop=crop))
            else:
                need = target - len(good) if target else len(failed_crops)
                print(f"  {split}: {failed} weather failures, retrying {need}...")
                records = gen._generate_more_records(split, need)

        if target:
            good = good[:target]
        if total_failed:
            print(f"  {split}: {total_failed} weather validations failed and replaced")
        validated[split] = good

    # ── Step 2: save per-split, per-crop pool ──
    # train/, val/, and test/ are physically separated so there is zero
    # overlap and evaluation splits do not depend on runtime sampling seeds.
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print("Weather pool built successfully!")
    print(f"{'=' * 60}")

    total = 0
    for split, configs in validated.items():
        pool_split = split
        split_dir = output_dir / pool_split
        split_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        seen: set[tuple] = set()
        for cfg in configs:
            ap = cfg.get("agro_params", {})
            crop = cfg.get("crop_name", "")
            year = int(ap.get("year", 0))
            lat = round(float(ap.get("latitude", 0)), 2)
            lon = round(float(ap.get("longitude", 0)), 2)
            key = (crop, year, lat, lon)
            if key in seen or not crop:
                continue
            seen.add(key)
            rows.append({
                "crop_name": crop,
                "year": year,
                "latitude": lat,
                "longitude": lon,
            })

        df = pd.DataFrame(rows)
        total += len(df)
        print(f"\n  {pool_split}/ ({len(df)} scenarios):")
        for crop, group in sorted(df.groupby("crop_name"), key=lambda x: x[0]):
            out = group[["year", "latitude", "longitude"]].reset_index(drop=True)
            path = split_dir / f"{crop}.parquet"
            out.to_parquet(path, index=False)
            print(f"    {crop}: {len(out)} entries")

    print(f"\n  Total: {total} scenarios")

    print(f"\nPool saved to: {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a WOFOST weather pool from a weather-pool generation config",
    )
    parser.add_argument(
        "config_id",
        help="Weather-pool config identifier (e.g. pool_4id_2ood). "
             "Reads from weather_pool_configs/<config_id>.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=SCRIPT_DIR / "weather_pool",
        help="Output directory for pool parquets (default: integrations/wofost_gym/dataset_tools/weather_pool)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override number of parallel workers (default: from config or 64)",
    )
    parser.add_argument(
        "--weather-cache-dir",
        type=Path,
        default=None,
        help="Use this PCSE meteo cache directory while validating generated weather.",
    )
    args = parser.parse_args()

    config_path = CONFIG_DIR / f"{args.config_id}.yaml"
    if not config_path.exists():
        available = [p.stem for p in CONFIG_DIR.glob("*.yaml")]
        parser.error(
            f"Config not found: {config_path}\n"
            f"Available configs: {', '.join(sorted(available))}"
        )

    print(f"{'=' * 60}")
    print("Weather Pool Generation")
    print(f"{'=' * 60}")
    print(f"Config: {config_path}")
    print(f"Output: {args.output_dir}")
    if args.num_workers:
        print(f"Workers: {args.num_workers}")
    if args.weather_cache_dir:
        print(f"Weather cache: {args.weather_cache_dir}")
    print(f"{'=' * 60}")

    build_pool(
        config_path,
        args.output_dir,
        num_workers=args.num_workers,
        weather_cache_dir=args.weather_cache_dir,
    )


if __name__ == "__main__":
    main()
