#!/usr/bin/env python3
"""Quick DVS maturity check: run no-action rollout across multiple weather scenarios.

For each crop, test several (latitude, longitude, year) combos to see how
weather variation affects DVS progression within 240 days.

Usage:
    python integrations/wofost_gym/crop_pool_design/check_crop_maturity.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from agrimanager.env.base import create_environment

CROPS = [
    "wheat", "maize", "sorghum", "sunflower",
    "millet", "potato",
    "barley", "rice", "tobacco", "soybean",
    "chickpea", "cotton", "cowpea", "fababean",
    "groundnut", "mungbean", "pigeonpea", "rapeseed",
    "seed_onion", "sugarbeet", "sweetpotato", "cassava",
]

INTERVAL = 10
STEPS = 24
TURN_NUM = STEPS

# Weather scenarios: (latitude, longitude, year) combos
# Cover a range of climates: northern temperate, southern temperate, subtropical, tropical
WEATHER_SCENARIOS = [
    # Northern Europe (cool)
    {"latitude": 52, "longitude": 5,   "year": 1990, "label": "NL-1990"},
    {"latitude": 52, "longitude": 5,   "year": 2005, "label": "NL-2005"},
    # Central Europe (temperate)
    {"latitude": 48, "longitude": 10,  "year": 1995, "label": "DE-1995"},
    {"latitude": 45, "longitude": 2,   "year": 2000, "label": "FR-2000"},
    # Southern Europe / Mediterranean (warm temperate)
    {"latitude": 40, "longitude": 15,  "year": 1998, "label": "IT-1998"},
    {"latitude": 38, "longitude": -5,  "year": 2003, "label": "ES-2003"},
    # US Midwest (continental)
    {"latitude": 42, "longitude": -93, "year": 2000, "label": "IA-2000"},
    {"latitude": 40, "longitude": -99, "year": 2010, "label": "KS-2010"},
    # Subtropical
    {"latitude": 30, "longitude": 5,   "year": 2000, "label": "N.Africa-2000"},
    {"latitude": 25, "longitude": 80,  "year": 2005, "label": "India-2005"},
    # Tropical
    {"latitude": 15, "longitude": 5,   "year": 2000, "label": "Sahel-2000"},
    {"latitude": 10, "longitude": 80,  "year": 2005, "label": "SriLanka-2005"},
]


def read_agro_defaults(crop_name: str) -> Dict[str, Any]:
    """Read the default agro file to get crop_start_type, crop_end_type, etc."""
    agro_dir = repo_root / ".." / "AgriManagerExternal" / "WOFOSTGym" / "env_config" / "agro"
    agro_path = agro_dir / f"{crop_name}_agro.yaml"
    with open(agro_path) as f:
        data = yaml.safe_load(f)
    return data["AgroManagement"]["CropCalendar"]


def make_agro_override(crop_name: str, scenario: Dict) -> Dict[str, Any]:
    """Build agro_params override for a weather scenario.

    Keeps original crop calendar structure but changes location/year.
    Uses a 270-day window (wider than 240) so the sim doesn't cut short.
    """
    defaults = read_agro_defaults(crop_name)

    lat = scenario["latitude"]
    lon = scenario["longitude"]
    year = scenario["year"]

    # Start date: pick a reasonable sowing month for the crop's base temperature
    # Cool crops (wheat, barley, fababean, rapeseed, potato, sugarbeet, seed_onion): Feb
    # Warm crops (maize, sorghum, millet, cotton, etc.): Apr-May
    cool_crops = {"wheat", "barley", "fababean", "rapeseed", "potato", "sugarbeet", "seed_onion"}
    if crop_name in cool_crops:
        start_month, start_day = 2, 1
    else:
        start_month, start_day = 4, 15

    start_date = f"{year + 1}-{start_month:02d}-{start_day:02d}"
    # End date: 270 days later to give enough room
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=270)
    end_date = end_dt.strftime("%Y-%m-%d")

    return {
        "latitude": lat,
        "longitude": lon,
        "year": year,
        "crop_start_date": start_date,
        "crop_end_date": end_date,
        "site_start_date": start_date,
        "site_end_date": end_date,
    }


def run_crop_scenario(crop_name: str, scenario: Dict) -> Tuple[Optional[float], Optional[int], str]:
    """Run one crop x scenario, return (final_dvs, maturity_day, error_or_empty)."""
    agro_override = make_agro_override(crop_name, scenario)

    env_config = {
        "env_id": "lnpkw-v0",
        "agro_file": f"{crop_name}_agro.yaml",
        "llm_mode": False,
        "intvn_interval": INTERVAL,
        "turn_num": TURN_NUM,
        "env_reward": "RewardWSODeltaWrapper",
        "agro_params": agro_override,
        "seed": 42,
    }
    try:
        env, config = create_environment("wofost_gym", env_config)
    except Exception as e:
        return None, None, str(e)

    output_vars = env.env.unwrapped.get_output_vars()
    dvs_idx = output_vars.index("DVS") if "DVS" in output_vars else None

    try:
        raw_obs, _ = env.reset()
    except Exception as e:
        return None, None, f"reset: {e}"

    dvs_val = float(raw_obs[dvs_idx]) if dvs_idx is not None else 0.0
    maturity_day = None

    for step in range(1, STEPS + 1):
        try:
            raw_obs, reward, done, info = env.step(0)
        except Exception:
            break

        if raw_obs is not None and dvs_idx is not None:
            new_dvs = float(raw_obs[dvs_idx])
            if new_dvs >= dvs_val * 0.5 or new_dvs > 0.1:
                dvs_val = new_dvs

        if dvs_val >= 2.0 and maturity_day is None:
            maturity_day = step * INTERVAL
        if done:
            break

    return dvs_val, maturity_day, ""


def main():
    n_scenarios = len(WEATHER_SCENARIOS)
    print(f"Running no-action DVS maturity check: {INTERVAL}-day interval, {STEPS} steps ({STEPS * INTERVAL} days)")
    print(f"Testing {len(CROPS)} crops x {n_scenarios} weather scenarios = {len(CROPS) * n_scenarios} runs\n")

    # Header
    labels = [s["label"] for s in WEATHER_SCENARIOS]
    hdr = f"{'Crop':<14}"
    for lb in labels:
        hdr += f" {lb:>14}"
    hdr += f"  {'mature%':>7} {'DVS_min':>7} {'DVS_med':>7} {'DVS_max':>7}"
    print(hdr)
    print("-" * len(hdr))

    summary = {}

    for crop in CROPS:
        row = f"{crop:<14}"
        dvs_list = []
        mat_list = []
        err_count = 0

        for scenario in WEATHER_SCENARIOS:
            final_dvs, mat_day, err = run_crop_scenario(crop, scenario)
            if err:
                row += f" {'ERR':>14}"
                err_count += 1
            elif mat_day:
                row += f" {'M@'+str(mat_day)+'d':>14}"
                dvs_list.append(final_dvs)
                mat_list.append(1)
            else:
                row += f" {final_dvs:>14.3f}"
                dvs_list.append(final_dvs)
                mat_list.append(0)

        if dvs_list:
            import numpy as np
            arr = np.array(dvs_list)
            pct_mature = 100.0 * sum(mat_list) / len(mat_list)
            row += f"  {pct_mature:>6.0f}% {arr.min():>7.3f} {np.median(arr):>7.3f} {arr.max():>7.3f}"
            summary[crop] = {
                "mature_pct": pct_mature,
                "dvs_min": float(arr.min()),
                "dvs_median": float(np.median(arr)),
                "dvs_max": float(arr.max()),
                "errors": err_count,
            }
        else:
            row += f"  {'ALL_ERR':>7}"
            summary[crop] = {"mature_pct": 0, "dvs_min": 0, "dvs_median": 0, "dvs_max": 0, "errors": err_count}

        print(row)

    # Sorted summary
    print(f"\n{'='*80}")
    print("SUMMARY (sorted by median DVS@240)")
    print(f"{'='*80}")
    print(f"{'Crop':<14} {'mature%':>7} {'DVS_min':>8} {'DVS_median':>10} {'DVS_max':>8} {'errors':>6} {'Verdict'}")
    print("-" * 80)
    for crop, s in sorted(summary.items(), key=lambda x: -x[1]["dvs_median"]):
        if s["errors"] == n_scenarios:
            verdict = "BROKEN"
        elif s["dvs_median"] >= 2.0:
            verdict = "OK (fast maturity)"
        elif s["dvs_median"] >= 1.8:
            verdict = "OK (close, likely OK with fertilizer)"
        elif s["dvs_median"] >= 1.5:
            verdict = "MARGINAL"
        else:
            verdict = "TOO SLOW"
        print(f"{crop:<14} {s['mature_pct']:>6.0f}% {s['dvs_min']:>8.3f} {s['dvs_median']:>10.3f} {s['dvs_max']:>8.3f} {s['errors']:>6}   {verdict}")


if __name__ == "__main__":
    main()
