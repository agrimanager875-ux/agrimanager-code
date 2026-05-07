#!/usr/bin/env python3
"""Build rice variety trait cards from WOFOST crop parameters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from agrimanager.env.wofost_gym.crop_trait_schemas import (
    RICE_VARIETY_TRAIT_SCHEMA,
    crop_variety_trait_key,
)


DEFAULT_OUTPUT_DIR = Path("agrimanager") / "env" / "wofost_gym" / "crop_traits"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _wofost_root() -> Path:
    return Path(
        os.environ.get(
            "WOFOST_GYM_PATH",
            str((_repo_root() / ".." / "AgriManagerExternal" / "WOFOSTGym").resolve()),
        )
    )


def _scalar(param: Any) -> float | None:
    if isinstance(param, (int, float)):
        return float(param)
    if isinstance(param, list) and param and isinstance(param[0], (int, float)):
        return float(param[0])
    return None


def _table_pairs(param: Any) -> list[tuple[float, float]]:
    if not isinstance(param, list) or not param or not isinstance(param[0], list):
        return []
    values = param[0]
    if len(values) < 2 or len(values) % 2 != 0:
        return []
    pairs: list[tuple[float, float]] = []
    for idx in range(0, len(values), 2):
        pairs.append((float(values[idx]), float(values[idx + 1])))
    return sorted(pairs)


def _max_y(param: Any) -> float | None:
    pairs = _table_pairs(param)
    if not pairs:
        return None
    return max(y for _, y in pairs)


def _first_x_with_y(param: Any, threshold: float) -> float | None:
    for x, y in _table_pairs(param):
        if y >= threshold:
            return x
    return None


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _card(crop: str, variety: str, params: dict[str, Any]) -> dict[str, Any]:
    tsum1 = _scalar(params.get("TSUM1"))
    tsum2 = _scalar(params.get("TSUM2"))
    tsum3 = _scalar(params.get("TSUM3"))
    tsum_total = sum(v for v in (tsum1, tsum2, tsum3) if v is not None)
    dlo = _scalar(params.get("DLO"))
    rgrlai = _scalar(params.get("RGRLAI"))
    tdwi = _scalar(params.get("TDWI"))
    vernbase = _scalar(params.get("VERNBASE"))
    vernsat = _scalar(params.get("VERNSAT"))
    amax_peak = _max_y(params.get("AMAXTB"))
    eff_peak = _max_y(params.get("EFFTB"))
    storage_onset = _first_x_with_y(params.get("FOTB"), 0.01)
    storage_peak = _max_y(params.get("FOTB"))

    return {
        "crop": crop,
        "variety": variety,
        "trait_key": crop_variety_trait_key(crop, variety),
        "trait_schema": RICE_VARIETY_TRAIT_SCHEMA,
        "core_facts": {
            "phenology": {
                "TSUM1_Cd": tsum1,
                "TSUM2_Cd": tsum2,
                "TSUM3_Cd": tsum3,
                "TSUM_total_Cd": tsum_total,
                "DLO_hr": dlo,
                "VERNBASE_d": vernbase,
                "VERNSAT_d": vernsat,
            },
            "growth": {
                "RGRLAI_per_day": rgrlai,
                "TDWI_kg_ha": tdwi,
            },
            "assimilation_and_partition": {
                "AMAX_peak": amax_peak,
                "EFF_peak": eff_peak,
                "FOTB_storage_onset_DVS": storage_onset,
                "FOTB_peak": storage_peak,
            },
        },
        "derived_traits": {
            "season_length": (
                "long"
                if tsum_total >= 2200
                else "medium"
                if tsum_total >= 1700
                else "short"
            ),
            "leaf_expansion": (
                "fast"
                if (rgrlai or 0.0) >= 0.009
                else "moderate"
                if (rgrlai or 0.0) >= 0.008
                else "slow"
            ),
            "storage_allocation": (
                "early"
                if storage_onset is not None and storage_onset <= 0.9
                else "mid"
            ),
        },
        "notes": [
            "This card is generated from WOFOST rice variety parameters only.",
            "Trait labels summarize simulator parameters, not field-trial cultivar guarantees.",
        ],
    }


def _text(card: dict[str, Any]) -> str:
    facts = card["core_facts"]
    phenology = facts["phenology"]
    growth = facts["growth"]
    assimilation = facts["assimilation_and_partition"]
    derived = card["derived_traits"]
    return "\n".join(
        [
            f"Crop Name: {card['crop']}",
            "",
            "Rice Variety Profile",
            (
                "- Phenology: "
                f"{derived['season_length']} season "
                f"(TSUM1={_fmt(phenology['TSUM1_Cd'], 0)} C.d, "
                f"TSUM2={_fmt(phenology['TSUM2_Cd'], 0)} C.d, "
                f"total={_fmt(phenology['TSUM_total_Cd'], 0)} C.d)"
            ),
            (
                "- Development cue: "
                f"DLO={_fmt(phenology['DLO_hr'], 1)} h, "
                f"vernalization base={_fmt(phenology['VERNBASE_d'], 1)} d, "
                f"saturation={_fmt(phenology['VERNSAT_d'], 1)} d"
            ),
            (
                "- Leaf expansion: "
                f"{derived['leaf_expansion']} "
                f"(RGRLAI={_fmt(growth['RGRLAI_per_day'], 4)} d-1)"
            ),
            (
                "- Initial biomass: "
                f"TDWI={_fmt(growth['TDWI_kg_ha'], 1)} kg ha-1"
            ),
            (
                "- Assimilation and storage: "
                f"AMAX peak={_fmt(assimilation['AMAX_peak'], 1)}, "
                f"EFF peak={_fmt(assimilation['EFF_peak'], 3)}, "
                f"storage onset DVS={_fmt(assimilation['FOTB_storage_onset_DVS'], 2)}"
            ),
            "",
            "Use guidance",
            "- Treat this as variety-level prior knowledge for rice management transfer.",
            "- Final action should still adapt to current weather, soil, nutrient, and growth observations.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = _repo_root() / output_root
    output_dir = output_root / RICE_VARIETY_TRAIT_SCHEMA
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_path = _wofost_root() / "env_config" / "crop" / "rice.yaml"
    crop_data = yaml.safe_load(crop_path.read_text(encoding="utf-8"))
    varieties = crop_data["CropParameters"]["Varieties"]

    for variety, params in sorted(varieties.items()):
        card = _card("rice", str(variety), params)
        trait_key = card["trait_key"]
        (output_dir / f"{trait_key}.json").write_text(
            json.dumps(card, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / f"{trait_key}.txt").write_text(_text(card), encoding="utf-8")

    print(f"Wrote {len(varieties)} rice variety trait cards to {output_dir}")


if __name__ == "__main__":
    main()
