#!/usr/bin/env python
"""Build a crop-trait schema from discovered strategy traits."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agrimanager.env.wofost_gym.crop_trait_schemas import (
    DEFAULT_CROP_TRAIT_SCHEMA,
    crop_variety_trait_key,
    resolve_crop_trait_schema_dir,
)
from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder

from diagnose_existing_specialist_transfer import ANALYSIS_DIR, DEFAULT_TRAITS_DIR, resolve_path
from discover_strategy_traits import derived_feature_values


DEFAULT_DISCOVERED_TRAITS = ANALYSIS_DIR / "strategy_trait_discovery" / "discovered_traits.json"
DEFAULT_SCHEMA_NAME = "traits_strategy_selected_v1"
DEFAULT_OUTPUT_SCHEMA_DIR = DEFAULT_TRAITS_DIR / DEFAULT_SCHEMA_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write selected strategy traits as a schema-aware crop-trait card directory."
    )
    parser.add_argument("--discovered-traits", type=Path, default=DEFAULT_DISCOVERED_TRAITS)
    parser.add_argument("--traits-dir", type=Path, default=DEFAULT_TRAITS_DIR)
    parser.add_argument("--source-trait-schema", default="")
    parser.add_argument("--schema-name", default=DEFAULT_SCHEMA_NAME)
    parser.add_argument("--output-schema-dir", type=Path, default=DEFAULT_OUTPUT_SCHEMA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_name(feature_name: str, rank: int) -> str:
    name = feature_name
    for prefix in ("core_facts.", "derived."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return f"t{rank:02d}_{name}"


def _load_source_cards(traits_dir: Path, source_trait_schema: str) -> dict[str, dict[str, Any]]:
    schema_dir = resolve_crop_trait_schema_dir(traits_dir, source_trait_schema)
    cards: dict[str, dict[str, Any]] = {}
    for path in sorted(schema_dir.glob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        crop = str(card.get("crop") or "").strip()
        variety = str(card.get("variety") or "").strip()
        trait_key = str(card.get("trait_key") or "").strip()
        if not trait_key:
            trait_key = crop_variety_trait_key(crop, variety) if crop and variety else (crop or path.stem)
        cards[trait_key] = card
    if not cards:
        raise FileNotFoundError(f"No source trait cards found in {schema_dir}")
    return cards


def _format_value(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def build_schema(
    *,
    discovered_traits: Path,
    traits_dir: Path,
    source_trait_schema: str,
    schema_name: str,
    output_schema_dir: Path,
    overwrite: bool,
) -> None:
    artifact = json.loads(discovered_traits.read_text(encoding="utf-8"))
    selected_features = [str(name) for name in artifact.get("selected_features", [])]
    if not selected_features:
        raise ValueError(f"No selected_features found in {discovered_traits}")
    source_schema = source_trait_schema or str(
        artifact.get("source_trait_schema") or DEFAULT_CROP_TRAIT_SCHEMA
    )

    if output_schema_dir.exists() and any(output_schema_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output schema directory is not empty: {output_schema_dir}. "
            "Pass --overwrite to replace generated cards."
        )
    output_schema_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_schema_dir.glob("*"):
            if path.is_file() and path.suffix in {".json", ".txt"}:
                path.unlink()

    encoder = CropTraitEncoder(traits_dir=traits_dir, trait_schema=source_schema)
    source_cards = _load_source_cards(traits_dir, source_schema)
    safe_names = {
        feature: _safe_name(feature, rank)
        for rank, feature in enumerate(selected_features, start=1)
    }

    manifest_rows = []
    for trait_key in encoder.crop_names:
        raw = encoder._raw_by_crop[trait_key]  # noqa: SLF001 - schema generation utility
        derived = derived_feature_values(raw)
        source_card = source_cards.get(trait_key, {})
        crop = str(source_card.get("crop") or trait_key.split("__", 1)[0]).strip()
        variety = str(source_card.get("variety") or "").strip()

        selected_values: dict[str, float] = {}
        original_values: dict[str, float | None] = {}
        for feature in selected_features:
            value = derived.get(feature) if feature.startswith("derived.") else raw.get(feature)
            if value is None or not math.isfinite(float(value)):
                value = 0.0
            selected_values[safe_names[feature]] = float(value)
            original_values[feature] = float(value)

        card = {
            "crop": crop,
            "variety": variety,
            "trait_key": trait_key,
            "trait_schema": schema_name,
            "core_facts": {
                "strategy_selected": selected_values,
            },
            "source_traits": {
                "source_trait_schema": source_schema,
                "selected_features": selected_features,
                "feature_name_map": safe_names,
                "raw_or_derived_values": original_values,
            },
            "notes": [
                "This card is generated from strategy-supervised specialist-transfer trait discovery.",
                "Traits are selected before downstream policy training and should be validated against shuffled controls.",
            ],
        }

        text_lines = [
            f"Crop Name: {crop}",
            "",
            "Strategy-Selected Traits",
        ]
        for feature in selected_features:
            safe = safe_names[feature]
            text_lines.append(
                f"- {safe}: {_format_value(selected_values[safe])} (source: {feature})"
            )
        text_lines.extend(
            [
                "",
                "Use guidance",
                "- Treat these as strategy-predictive crop descriptors, not crop identifiers.",
                "- Combine them with current weather, soil, and growth state before acting.",
            ]
        )
        output_keys = [trait_key]
        if crop and crop != trait_key:
            output_keys.append(crop)
        for output_key in output_keys:
            output_card = dict(card)
            output_card["trait_key"] = output_key
            json_path = output_schema_dir / f"{output_key}.json"
            json_path.write_text(
                json.dumps(output_card, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_schema_dir / f"{output_key}.txt").write_text(
                "\n".join(text_lines) + "\n",
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "trait_key": output_key,
                    "source_trait_key": trait_key,
                    "crop": crop,
                    "variety": variety,
                    "json": str(json_path),
                }
            )

    manifest = {
        "schema_name": schema_name,
        "source_trait_schema": source_schema,
        "discovered_traits": str(discovered_traits),
        "selected_features": selected_features,
        "feature_name_map": safe_names,
        "cards": manifest_rows,
    }
    (output_schema_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    build_schema(
        discovered_traits=resolve_path(args.discovered_traits),
        traits_dir=resolve_path(args.traits_dir),
        source_trait_schema=str(args.source_trait_schema or ""),
        schema_name=str(args.schema_name),
        output_schema_dir=resolve_path(args.output_schema_dir),
        overwrite=bool(args.overwrite),
    )
    print(f"wrote {resolve_path(args.output_schema_dir)}")


if __name__ == "__main__":
    main()
