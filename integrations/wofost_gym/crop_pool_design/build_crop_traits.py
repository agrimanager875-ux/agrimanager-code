#!/usr/bin/env python3
"""Build crop trait cards from WOFOST crop/agro configuration files.

This script now emits two schema variants:

- ``traits_v1_23d``: raw physiological facts used by the existing numeric trait
  encoder.
- ``traits_v1_6d``: compressed decision-facing axes intended to preserve
  agronomic guidance while reducing trait dimensionality.

The legacy flat files under ``agrimanager/env/wofost_gym/crop_traits/`` remain
the canonical 23D cards for backward compatibility. Schema-aware cards are also
written under ``<output_dir>/<schema>/``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from agrimanager.env.wofost_gym.crop_trait_schemas import (
    DEFAULT_CROP_TRAIT_SCHEMA,
    POLICY_CROP_TRAIT_SCHEMA,
)


@dataclass
class CropTraitsPaths:
    crop_dir: Path
    agro_dir: Path
    output_dir: Path


DEFAULT_OUTPUT_DIR = Path("agrimanager") / "env" / "wofost_gym" / "crop_traits"


def _resolve_paths(output_dir: Optional[str]) -> CropTraitsPaths:
    repo_root = Path(__file__).resolve().parents[4]
    wofost_root = Path(
        os.environ.get(
            "WOFOST_GYM_PATH",
            str((repo_root / ".." / "AgriManagerExternal" / "WOFOSTGym").resolve()),
        )
    )

    crop_dir = wofost_root / "env_config" / "crop"
    agro_dir = wofost_root / "env_config" / "agro"

    if output_dir:
        out = Path(output_dir)
        if not out.is_absolute():
            out = repo_root / out
    else:
        out = repo_root / DEFAULT_OUTPUT_DIR

    return CropTraitsPaths(crop_dir=crop_dir, agro_dir=agro_dir, output_dir=out)


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scalar(param: Any) -> Optional[float]:
    if isinstance(param, (int, float)):
        return float(param)
    if isinstance(param, list) and param:
        first = param[0]
        if isinstance(first, (int, float)):
            return float(first)
    return None


def _table_pairs(param: Any) -> List[Tuple[float, float]]:
    if not isinstance(param, list) or not param:
        return []
    first = param[0]
    if not isinstance(first, list):
        return []
    values = first
    if len(values) < 2 or len(values) % 2 != 0:
        return []

    pairs: List[Tuple[float, float]] = []
    for i in range(0, len(values), 2):
        try:
            x = float(values[i])
            y = float(values[i + 1])
            pairs.append((x, y))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda item: item[0])
    return pairs


def _interp(pairs: List[Tuple[float, float]], x: float) -> Optional[float]:
    if not pairs:
        return None
    if x <= pairs[0][0]:
        return pairs[0][1]
    if x >= pairs[-1][0]:
        return pairs[-1][1]

    for i in range(1, len(pairs)):
        x0, y0 = pairs[i - 1]
        x1, y1 = pairs[i]
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return pairs[-1][1]


def _first_x_with_y_at_least(
    pairs: List[Tuple[float, float]],
    threshold: float,
) -> Optional[float]:
    for x, y in pairs:
        if y >= threshold:
            return x
    return None


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_map(
    values: Dict[str, Optional[float]],
    *,
    invert: bool = False,
    missing_fill: float = 0.5,
) -> Dict[str, float]:
    finite = [value for value in values.values() if value is not None]
    if not finite:
        return {crop: float(missing_fill) for crop in values}

    lo = min(finite)
    hi = max(finite)
    span = hi - lo
    if span == 0.0:
        normalized = {crop: 0.5 for crop in values}
    else:
        normalized = {
            crop: (
                missing_fill
                if value is None
                else (float(value) - lo) / span
            )
            for crop, value in values.items()
        }

    if invert:
        return {crop: _clip01(1.0 - score) for crop, score in normalized.items()}
    return {crop: _clip01(score) for crop, score in normalized.items()}


def _label_thermal(total_tsum: Optional[float]) -> str:
    if total_tsum is None:
        return "unknown"
    if total_tsum >= 2300:
        return "long-season"
    if total_tsum >= 1700:
        return "medium-season"
    return "short-season"


def _label_temperature_base(tbase: Optional[float]) -> str:
    if tbase is None:
        return "unknown"
    if tbase <= 3:
        return "cool-adapted"
    if tbase <= 8:
        return "temperate-adapted"
    return "warm-adapted"


def _label_root_depth(rdmcr: Optional[float]) -> str:
    if rdmcr is None:
        return "unknown"
    if rdmcr >= 150:
        return "deep-rooted"
    if rdmcr >= 100:
        return "moderately deep-rooted"
    return "shallow-rooted"


def _label_water_stress(depnr: Optional[float]) -> str:
    if depnr is None:
        return "unknown"
    if depnr >= 4.5:
        return "more drought-tolerant"
    if depnr >= 3.5:
        return "moderately drought-tolerant"
    return "more drought-sensitive"


def _label_daylength(idsl: Optional[float]) -> str:
    if idsl is None:
        return "unknown"
    if idsl >= 2:
        return "temperature + daylength + vernalization"
    if idsl >= 1:
        return "temperature + daylength"
    return "temperature only"


def _label_storage_onset(fotb_onset_dvs: Optional[float]) -> str:
    if fotb_onset_dvs is None:
        return "unknown"
    if fotb_onset_dvs <= 0.9:
        return "early storage allocation"
    if fotb_onset_dvs <= 1.2:
        return "mid storage allocation"
    return "late storage allocation"


def _label_amax(amax_peak: Optional[float]) -> str:
    if amax_peak is None:
        return "unknown"
    if amax_peak >= 70:
        return "high assimilation potential"
    if amax_peak >= 40:
        return "moderate assimilation potential"
    return "lower assimilation potential"


def _decision_level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def _extract_crop_variety(agro_data: Dict[str, Any]) -> str:
    try:
        return str(agro_data["AgroManagement"]["CropCalendar"]["crop_variety"])
    except KeyError as exc:
        raise KeyError("Could not read AgroManagement.CropCalendar.crop_variety") from exc


def _extract_variety_params(crop_data: Dict[str, Any], variety: str) -> Dict[str, Any]:
    try:
        varieties = crop_data["CropParameters"]["Varieties"]
    except KeyError as exc:
        raise KeyError("CropParameters.Varieties not found") from exc

    if variety not in varieties:
        known = ", ".join(sorted(varieties.keys())[:10])
        raise KeyError(f"Variety '{variety}' not found. Known examples: {known}")

    return varieties[variety]


def _build_structured_card(crop: str, variety: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # Scalars
    tbase = _scalar(params.get("TBASEM"))
    teffmx = _scalar(params.get("TEFFMX"))
    tsumem = _scalar(params.get("TSUMEM"))
    tsum1 = _scalar(params.get("TSUM1"))
    tsum2 = _scalar(params.get("TSUM2"))
    tsum3 = _scalar(params.get("TSUM3"))
    idsl = _scalar(params.get("IDSL"))
    dlo = _scalar(params.get("DLO"))
    dlc = _scalar(params.get("DLC"))
    rdmcr = _scalar(params.get("RDMCR"))
    rri = _scalar(params.get("RRI"))
    depnr = _scalar(params.get("DEPNR"))
    cfet = _scalar(params.get("CFET"))
    iairdu = _scalar(params.get("IAIRDU"))

    total_tsum = None
    if tsumem is not None and tsum1 is not None and tsum2 is not None:
        total_tsum = tsumem + tsum1 + tsum2

    # Tables
    nmax_pairs = _table_pairs(params.get("NMAXLV_TB"))
    pmax_pairs = _table_pairs(params.get("PMAXLV_TB"))
    kmax_pairs = _table_pairs(params.get("KMAXLV_TB"))
    amax_pairs = _table_pairs(params.get("AMAXTB"))
    fotb_pairs = _table_pairs(params.get("FOTB"))

    n_dvs0 = _interp(nmax_pairs, 0.0)
    n_dvs1 = _interp(nmax_pairs, 1.0)
    p_dvs0 = _interp(pmax_pairs, 0.0)
    p_dvs1 = _interp(pmax_pairs, 1.0)
    k_dvs0 = _interp(kmax_pairs, 0.0)
    k_dvs1 = _interp(kmax_pairs, 1.0)

    amax_peak = max((y for _, y in amax_pairs), default=None)
    storage_onset_dvs = _first_x_with_y_at_least(fotb_pairs, 0.1)

    return {
        "crop": crop,
        "variety": variety,
        "trait_schema": DEFAULT_CROP_TRAIT_SCHEMA,
        "core_facts": {
            "temperature": {
                "TBASEM_C": tbase,
                "TEFFMX_C": teffmx,
            },
            "phenology": {
                "TSUMEM_Cd": tsumem,
                "TSUM1_Cd": tsum1,
                "TSUM2_Cd": tsum2,
                "TSUM3_Cd": tsum3,
                "TSUM_total_Cd": total_tsum,
                "IDSL": idsl,
                "DLO_hr": dlo,
                "DLC_hr": dlc,
            },
            "root_and_water": {
                "RDMCR_cm": rdmcr,
                "RRI_cm_per_day": rri,
                "DEPNR": depnr,
                "CFET": cfet,
                "IAIRDU": iairdu,
            },
            "nutrient_capacity_leaf": {
                "NMAXLV_DVS0": n_dvs0,
                "NMAXLV_DVS1": n_dvs1,
                "PMAXLV_DVS0": p_dvs0,
                "PMAXLV_DVS1": p_dvs1,
                "KMAXLV_DVS0": k_dvs0,
                "KMAXLV_DVS1": k_dvs1,
            },
            "assimilation_and_partition": {
                "AMAX_peak": amax_peak,
                "FOTB_storage_onset_DVS": storage_onset_dvs,
            },
        },
        "derived_traits": {
            "season_length_class": _label_thermal(total_tsum),
            "temperature_adaptation": _label_temperature_base(tbase),
            "rooting_class": _label_root_depth(rdmcr),
            "water_stress_response": _label_water_stress(depnr),
            "development_driver": _label_daylength(idsl),
            "storage_allocation_timing": _label_storage_onset(storage_onset_dvs),
            "assimilation_class": _label_amax(amax_peak),
        },
        "notes": [
            "This card is generated from WOFOST parameter files only.",
            "Trait labels are heuristic summaries of model parameters, not field trial guarantees.",
        ],
    }


def _render_card_text(card: Dict[str, Any]) -> str:
    crop = card["crop"]
    core = card["core_facts"]
    derived = card["derived_traits"]

    phen = core["phenology"]
    temp = core["temperature"]
    rw = core["root_and_water"]
    nut = core["nutrient_capacity_leaf"]
    ap = core["assimilation_and_partition"]

    lines = [
        f"Crop Name: {crop}",
        "",
        "Profile",
        f"- Season type: {derived['season_length_class']} (TSUM total={_fmt(phen['TSUM_total_Cd'], 0)} C.d)",
        f"- Temperature adaptation: {derived['temperature_adaptation']} (TBASEM={_fmt(temp['TBASEM_C'])} C)",
        f"- Development driver: {derived['development_driver']} (IDSL={_fmt(phen['IDSL'], 0)}, DLO={_fmt(phen['DLO_hr'])} h)",
        f"- Root/water trait: {derived['rooting_class']}, {derived['water_stress_response']} (RDMCR={_fmt(rw['RDMCR_cm'])} cm, DEPNR={_fmt(rw['DEPNR'])})",
        f"- Assimilation trait: {derived['assimilation_class']} (AMAX peak={_fmt(ap['AMAX_peak'])})",
        f"- Storage allocation: {derived['storage_allocation_timing']} (FOTB onset DVS={_fmt(ap['FOTB_storage_onset_DVS'])})",
        "",
        "Nutrient capacity proxies (leaf maxima)",
        f"- NMAXLV: DVS0={_fmt(nut['NMAXLV_DVS0'], 4)}, DVS1={_fmt(nut['NMAXLV_DVS1'], 4)}",
        f"- PMAXLV: DVS0={_fmt(nut['PMAXLV_DVS0'], 4)}, DVS1={_fmt(nut['PMAXLV_DVS1'], 4)}",
        f"- KMAXLV: DVS0={_fmt(nut['KMAXLV_DVS0'], 4)}, DVS1={_fmt(nut['KMAXLV_DVS1'], 4)}",
        "",
        "Use guidance",
        "- Treat this as prior knowledge for zero-shot policy transfer.",
        "- Final action should still adapt to current weather/soil observations.",
    ]
    return "\n".join(lines)


def _build_policy_cards(cards_23d: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    total_tsum = {
        crop: card["core_facts"]["phenology"]["TSUM_total_Cd"]
        for crop, card in cards_23d.items()
    }
    fotb = {
        crop: card["core_facts"]["assimilation_and_partition"]["FOTB_storage_onset_DVS"]
        for crop, card in cards_23d.items()
    }
    tbase = {
        crop: card["core_facts"]["temperature"]["TBASEM_C"]
        for crop, card in cards_23d.items()
    }
    idsl = {
        crop: card["core_facts"]["phenology"]["IDSL"]
        for crop, card in cards_23d.items()
    }
    depnr = {
        crop: card["core_facts"]["root_and_water"]["DEPNR"]
        for crop, card in cards_23d.items()
    }
    rdmcr = {
        crop: card["core_facts"]["root_and_water"]["RDMCR_cm"]
        for crop, card in cards_23d.items()
    }
    cfet = {
        crop: card["core_facts"]["root_and_water"]["CFET"]
        for crop, card in cards_23d.items()
    }
    iairdu = {
        crop: card["core_facts"]["root_and_water"]["IAIRDU"]
        for crop, card in cards_23d.items()
    }
    n_dvs0 = {
        crop: card["core_facts"]["nutrient_capacity_leaf"]["NMAXLV_DVS0"]
        for crop, card in cards_23d.items()
    }
    n_dvs1 = {
        crop: card["core_facts"]["nutrient_capacity_leaf"]["NMAXLV_DVS1"]
        for crop, card in cards_23d.items()
    }
    p_dvs0 = {
        crop: card["core_facts"]["nutrient_capacity_leaf"]["PMAXLV_DVS0"]
        for crop, card in cards_23d.items()
    }
    p_dvs1 = {
        crop: card["core_facts"]["nutrient_capacity_leaf"]["PMAXLV_DVS1"]
        for crop, card in cards_23d.items()
    }
    k_dvs0 = {
        crop: card["core_facts"]["nutrient_capacity_leaf"]["KMAXLV_DVS0"]
        for crop, card in cards_23d.items()
    }
    k_dvs1 = {
        crop: card["core_facts"]["nutrient_capacity_leaf"]["KMAXLV_DVS1"]
        for crop, card in cards_23d.items()
    }

    urgency_tsum = _normalize_map(total_tsum, invert=True)
    urgency_fotb = _normalize_map(fotb, invert=True)
    constraint_tbase = _normalize_map(tbase)
    constraint_idsl = _normalize_map(idsl)
    water_sensitivity = _normalize_map(depnr, invert=True)
    water_root = _normalize_map(rdmcr, invert=True)
    water_cfet = _normalize_map(cfet)
    water_air = _normalize_map(iairdu)
    n0_norm = _normalize_map(n_dvs0)
    n1_norm = _normalize_map(n_dvs1)
    p0_norm = _normalize_map(p_dvs0)
    p1_norm = _normalize_map(p_dvs1)
    k0_norm = _normalize_map(k_dvs0)
    k1_norm = _normalize_map(k_dvs1)

    cards_6d: Dict[str, Dict[str, Any]] = {}
    for crop, base_card in cards_23d.items():
        season_urgency = _clip01(0.65 * urgency_tsum[crop] + 0.35 * urgency_fotb[crop])
        development_constraint = _clip01(
            0.55 * constraint_tbase[crop] + 0.45 * constraint_idsl[crop]
        )
        water_priority = _clip01(
            0.35 * water_sensitivity[crop]
            + 0.25 * water_root[crop]
            + 0.20 * water_air[crop]
            + 0.20 * water_cfet[crop]
        )
        nitrogen_priority = _clip01(0.70 * n0_norm[crop] + 0.30 * n1_norm[crop])
        phosphorus_priority = _clip01(0.70 * p0_norm[crop] + 0.30 * p1_norm[crop])
        potassium_priority = _clip01(0.70 * k0_norm[crop] + 0.30 * k1_norm[crop])

        cards_6d[crop] = {
            "crop": crop,
            "variety": base_card["variety"],
            "trait_schema": POLICY_CROP_TRAIT_SCHEMA,
            "core_facts": {
                "decision_axes": {
                    "season_urgency": season_urgency,
                    "development_constraint": development_constraint,
                    "water_priority": water_priority,
                    "nitrogen_priority": nitrogen_priority,
                    "phosphorus_priority": phosphorus_priority,
                    "potassium_priority": potassium_priority,
                }
            },
            "source_facts": {
                "season_urgency": {
                    "TSUM_total_Cd": total_tsum[crop],
                    "FOTB_storage_onset_DVS": fotb[crop],
                },
                "development_constraint": {
                    "TBASEM_C": tbase[crop],
                    "IDSL": idsl[crop],
                },
                "water_priority": {
                    "DEPNR": depnr[crop],
                    "RDMCR_cm": rdmcr[crop],
                    "CFET": cfet[crop],
                    "IAIRDU": iairdu[crop],
                },
                "nutrient_priority": {
                    "NMAXLV_DVS0": n_dvs0[crop],
                    "NMAXLV_DVS1": n_dvs1[crop],
                    "PMAXLV_DVS0": p_dvs0[crop],
                    "PMAXLV_DVS1": p_dvs1[crop],
                    "KMAXLV_DVS0": k_dvs0[crop],
                    "KMAXLV_DVS1": k_dvs1[crop],
                },
            },
            "derived_traits": {
                "season_urgency_level": _decision_level(season_urgency),
                "development_constraint_level": _decision_level(development_constraint),
                "water_priority_level": _decision_level(water_priority),
                "nitrogen_priority_level": _decision_level(nitrogen_priority),
                "phosphorus_priority_level": _decision_level(phosphorus_priority),
                "potassium_priority_level": _decision_level(potassium_priority),
            },
            "notes": [
                "This 6D card is derived from the raw WOFOST parameter card for policy transfer.",
                "The axes are heuristic decision-facing summaries, not causal guarantees.",
            ],
        }

    return cards_6d


def _render_policy_card_text(card: Dict[str, Any]) -> str:
    crop = card["crop"]
    axes = card["core_facts"]["decision_axes"]
    levels = card["derived_traits"]

    lines = [
        f"Crop Name: {crop}",
        "",
        "Decision Profile",
        (
            "- Season urgency: "
            f"{levels['season_urgency_level']} (score={axes['season_urgency']:.2f}) "
            "Higher means shorter effective intervention windows and lower payoff from late corrections."
        ),
        (
            "- Development constraint: "
            f"{levels['development_constraint_level']} (score={axes['development_constraint']:.2f}) "
            "Higher means early growth can be more thermally or photoperiod constrained; avoid overreacting to slow early biomass alone."
        ),
        (
            "- Water priority: "
            f"{levels['water_priority_level']} (score={axes['water_priority']:.2f}) "
            "Higher means emerging water stress should more often push irrigation ahead of extra fertilizer."
        ),
        (
            "- Nitrogen priority: "
            f"{levels['nitrogen_priority_level']} (score={axes['nitrogen_priority']:.2f}) "
            "Higher means low nutrient status is more likely to justify nitrogen correction."
        ),
        (
            "- Phosphorus priority: "
            f"{levels['phosphorus_priority_level']} (score={axes['phosphorus_priority']:.2f}) "
            "Higher means phosphorus limitation is more likely to matter for this crop."
        ),
        (
            "- Potassium priority: "
            f"{levels['potassium_priority_level']} (score={axes['potassium_priority']:.2f}) "
            "Higher means potassium correction is more likely to have value when nutrient status is low."
        ),
        "",
        "Use guidance",
        "- Treat these axes as a compact decision prior rather than a crop identifier.",
        "- Combine them with current weather, soil, and growth observations before choosing an action.",
    ]
    return "\n".join(lines)


def _write_card_pair(
    output_dir: Path,
    crop: str,
    card: Dict[str, Any],
    card_text: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{crop}.json"
    txt_path = output_dir / f"{crop}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(card_text)
        f.write("\n")

    return [json_path, txt_path]


def _discover_available_crops(paths: CropTraitsPaths) -> List[str]:
    crops = []
    for agro_path in sorted(paths.agro_dir.glob("*_agro.yaml")):
        crop = agro_path.name[:-10]
        if (paths.crop_dir / f"{crop}.yaml").exists():
            crops.append(crop)
    return crops


def build_cards(crops: Iterable[str], paths: CropTraitsPaths) -> List[Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    selected_crops = list(dict.fromkeys(crops))
    cards_23d: Dict[str, Dict[str, Any]] = {}

    for crop in selected_crops:
        crop_yaml = paths.crop_dir / f"{crop}.yaml"
        agro_yaml = paths.agro_dir / f"{crop}_agro.yaml"
        if not crop_yaml.exists():
            raise FileNotFoundError(f"Missing crop file: {crop_yaml}")
        if not agro_yaml.exists():
            raise FileNotFoundError(f"Missing agro file: {agro_yaml}")

        crop_data = _safe_load_yaml(crop_yaml)
        agro_data = _safe_load_yaml(agro_yaml)
        variety = _extract_crop_variety(agro_data)
        params = _extract_variety_params(crop_data, variety)
        cards_23d[crop] = _build_structured_card(crop=crop, variety=variety, params=params)

    cards_6d = _build_policy_cards(cards_23d)

    written: List[Path] = []
    schema_23d_dir = paths.output_dir / DEFAULT_CROP_TRAIT_SCHEMA
    schema_6d_dir = paths.output_dir / POLICY_CROP_TRAIT_SCHEMA

    for crop in selected_crops:
        card_23d = cards_23d[crop]
        card_6d = cards_6d[crop]
        text_23d = _render_card_text(card_23d)
        text_6d = _render_policy_card_text(card_6d)

        # Keep the legacy flat layout as the canonical 23D default.
        written.extend(_write_card_pair(paths.output_dir, crop, card_23d, text_23d))
        written.extend(_write_card_pair(schema_23d_dir, crop, card_23d, text_23d))
        written.extend(_write_card_pair(schema_6d_dir, crop, card_6d, text_6d))

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crop traits from WOFOST config files.")
    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        help=(
            "Crop names (matching both crop/<name>.yaml and agro/<name>_agro.yaml). "
            "Defaults to all crops that have both files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory (relative to repo root if not absolute). "
            "Defaults to agrimanager/env/wofost_gym/crop_traits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolve_paths(args.output_dir)
    crops = args.crops or _discover_available_crops(paths)
    written = build_cards(crops, paths)

    print("Generated crop traits:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
