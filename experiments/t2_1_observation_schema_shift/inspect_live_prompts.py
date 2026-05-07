#!/usr/bin/env python3
"""Render one-turn message comparisons for T2.1 observation schema shift.

This helper fixes a single shared WOFOST scenario from the unified T2.1 config,
then renders the first-turn chat messages for each active schema variant:

- S1 full current schema
- S2a/S2b/S2c/S2d component-drop variants
- S3 semantic synonym rename
- S4 compact growth superset
- S5 anonymous label rename

The report is prompt inspection only; no model inference is performed.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from agrimanager.env.base.utils import create_environment
from agrimanager.env.wofost_gym.create_dataset import WOFOSTArtifactDatasetBuilder
from agrimanager.env.wofost_gym.prompt import WOFOSTPromptGenerator


OUTPUT_DIR = REPO_ROOT / "experiments" / "t2_1_observation_schema_shift" / "prompt_analysis"
BUILDER_OUTPUT_ROOT = Path("/tmp/t21_observation_schema_shift_prompt_analysis")
LOCAL_WEATHER_POOL_DIR = (
    REPO_ROOT / ".cache" / "wofost_gym" / "weather_pool_20crop_3200_val128_test512"
)
ROOT_CONFIGS = {
    "think": REPO_ROOT
    / "experiments/t2_1_observation_schema_shift/config/t21_observation_schema_shift_llm_think.yaml",
    "no_think": REPO_ROOT
    / "experiments/t2_1_observation_schema_shift/config/t21_observation_schema_shift_llm_no_think.yaml",
}


def _default_output(mode: str) -> Path:
    return OUTPUT_DIR / f"live_first_turn_messages_{mode}.md"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    source_cfg = config.setdefault("source", {})
    if source_cfg.get("kind") == "weather_pool" and "local_cache_dir" not in source_cfg:
        source_cfg["local_cache_dir"] = str(LOCAL_WEATHER_POOL_DIR)
    env_cfg = config.setdefault("env", {})
    env_wofost_path = os.environ.get("WOFOST_GYM_PATH")
    if env_wofost_path:
        env_cfg["wofost_gym_path"] = env_wofost_path
    return config


def _make_builder(config_path: Path) -> Tuple[Dict[str, Any], WOFOSTArtifactDatasetBuilder]:
    config = _load_yaml(config_path)
    builder = WOFOSTArtifactDatasetBuilder(config, str(BUILDER_OUTPUT_ROOT))
    return config, builder


def _schema_split_order(split: str) -> List[str]:
    return [
        f"{split}_s1",
        f"{split}_s2a",
        f"{split}_s2b",
        f"{split}_s2c",
        f"{split}_s2d",
        f"{split}_s3",
        f"{split}_s4",
        f"{split}_s5",
    ]


def _shared_capture_split(split: str) -> str:
    return f"{split}_s4"


def _shared_scenario_record(
    builder: WOFOSTArtifactDatasetBuilder,
    split: str,
    scenario_index: int,
) -> Dict[str, Any]:
    capture_split = _shared_capture_split(split)
    records = builder.build_split_records(capture_split)
    if scenario_index < 0 or scenario_index >= len(records):
        raise IndexError(
            f"scenario_index={scenario_index} is out of range for split '{capture_split}' "
            f"with {len(records)} records."
        )
    return deepcopy(records[scenario_index])


def _schema_env_config(builder: WOFOSTArtifactDatasetBuilder, split_name: str) -> Dict[str, Any]:
    split_cfg = builder.sampling_cfg["splits"][split_name]
    return builder._materialize_split_env_config(split_cfg)


def _capture_shared_state(env_config: Dict[str, Any]):
    env, _ = create_environment("wofost_gym", deepcopy(env_config))
    _, info = env.reset()
    return env, info


def _render_prompt_from_shared_state(
    shared_env,
    env_config: Dict[str, Any],
    shared_prompt_observation: Dict[str, float],
) -> Tuple[str, str]:
    fields = list(env_config.get("prompt_observation_fields") or [])
    observation = [shared_prompt_observation[field] for field in fields]
    generator = WOFOSTPromptGenerator.from_env(
        shared_env.env,
        require_think=bool(env_config.get("require_think", False)),
        thinking_mode=str(env_config.get("thinking_mode", "grounding_decision")),
        think_tag=str(env_config.get("think_tag", "tool_call")),
        output_vars=fields,
        field_aliases=env_config.get("prompt_field_aliases"),
    )
    return generator.get_system_prompt(), generator.get_turn_prompt(observation)


def _ordered_difference(items: Iterable[str], to_remove: Iterable[str]) -> List[str]:
    remove = set(to_remove)
    return [item for item in items if item not in remove]


def _format_mapping_lines(
    mapping: Dict[str, str],
    baseline_aliases: Optional[Dict[str, str]] = None,
) -> List[str]:
    baseline_aliases = baseline_aliases or {}
    return [f"- `{key}`: `{baseline_aliases.get(key, key)}` -> `{value}`" for key, value in mapping.items()]


def _summarize_changes(
    baseline_fields: List[str],
    baseline_aliases: Dict[str, str],
    env_config: Dict[str, Any],
) -> List[str]:
    fields = list(env_config.get("prompt_observation_fields") or [])
    aliases = dict(env_config.get("prompt_field_aliases") or {})
    missing = [field for field in baseline_fields if field not in fields]
    added = [field for field in fields if field not in baseline_fields]
    renamed = {
        key: value
        for key, value in aliases.items()
        if baseline_aliases.get(key, key) != value
    }

    summary = [f"- Prompt fields ({len(fields)}): `{', '.join(fields)}`"]
    if missing:
        summary.append(f"- Missing baseline fields: `{', '.join(missing)}`")
    if added:
        summary.append(f"- Added fields: `{', '.join(added)}`")
    if renamed:
        summary.append("- Renamed prompt labels:")
        summary.extend(_format_mapping_lines(renamed, baseline_aliases=baseline_aliases))
    if not missing and not added and not renamed:
        summary.append("- Schema delta: none (baseline prompt layout).")
    return summary


def _render_message_block(system_prompt: str, user_prompt: str) -> List[str]:
    return [
        "### Messages",
        "",
        "#### System",
        "",
        "```text",
        system_prompt.rstrip(),
        "```",
        "",
        "#### User",
        "",
        "```text",
        user_prompt.rstrip(),
        "```",
        "",
    ]


def _render_schema_section(
    title: str,
    baseline_fields: List[str],
    baseline_aliases: Dict[str, str],
    env_config: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> str:
    lines = [f"## {title}", ""]
    schema_name = env_config.get("observation_schema_name")
    if schema_name:
        lines.append(f"- Observation schema: `{schema_name}`")
    lines.extend(_summarize_changes(baseline_fields, baseline_aliases, env_config))
    lines.append("")
    lines.extend(_render_message_block(system_prompt, user_prompt))
    return "\n".join(lines)


def _report_header(
    mode: str,
    split: str,
    scenario_index: int,
    shared_record: Dict[str, Any],
    shared_prompt_observation: Dict[str, float],
) -> str:
    agro_params = shared_record.get("agro_params") or {}
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    crop_name = shared_record.get("crop_name", "unknown")
    year = agro_params.get("year", shared_record.get("year"))
    latitude = agro_params.get("latitude")
    longitude = agro_params.get("longitude")
    scenario_id = shared_record.get("scenario_id", "unknown")
    paired_scenario_id = shared_record.get("paired_scenario_id", "unknown")
    seed = shared_record.get("seed", "unknown")

    lines = [
        "# T2.1 Live Message Analysis",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Prompt mode: `{mode}`",
        f"- Shared split family: `{split}`",
        f"- Shared scenario index: `{scenario_index}`",
        f"- Crop: `{crop_name}`",
        f"- Year: `{year}`",
        f"- Latitude / longitude: `{latitude}`, `{longitude}`",
        f"- Shared scenario_id: `{scenario_id}`",
        f"- Shared paired_scenario_id: `{paired_scenario_id}`",
        f"- Shared env seed: `{seed}`",
        "",
        "## Shared initial observation snapshot",
        "",
        f"`{', '.join(f'{key}={value:.4g}' for key, value in shared_prompt_observation.items())}`",
        "",
        "The sections below keep the simulator state fixed and only change the chat messages implied by the observation schema.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(ROOT_CONFIGS),
        default="think",
        help="Use the think or no_think unified T2.1 config.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Use the validation or test shared scenario family.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Zero-based scenario index inside the shared scenario family.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the markdown report.",
    )
    args = parser.parse_args()

    config_path = ROOT_CONFIGS[args.mode]
    config, builder = _make_builder(config_path)

    shared_record = _shared_scenario_record(builder, args.split, args.scenario_index)
    baseline_fields = list(
        config["sampling"]["splits"][f"{args.split}_s1"]["env"]["prompt_observation_fields"]
    )
    baseline_aliases = {field: field for field in baseline_fields}

    shared_env, shared_info = _capture_shared_state(shared_record)
    try:
        shared_prompt_observation = {
            key: float(value)
            for key, value in (shared_info.get("prompt_observation") or {}).items()
        }

        rendered_sections: List[str] = []
        for split_name in _schema_split_order(args.split):
            env_config = _schema_env_config(builder, split_name)
            system_prompt, user_prompt = _render_prompt_from_shared_state(
                shared_env,
                env_config,
                shared_prompt_observation,
            )
            rendered_sections.append(
                _render_schema_section(
                    title=str(env_config.get("observation_schema_name", split_name)),
                    baseline_fields=baseline_fields,
                    baseline_aliases=baseline_aliases,
                    env_config=env_config,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )

        report_text = _report_header(
            mode=args.mode,
            split=args.split,
            scenario_index=args.scenario_index,
            shared_record=shared_record,
            shared_prompt_observation=shared_prompt_observation,
        ) + "\n".join(rendered_sections)
    finally:
        shared_env.close()

    output_path = args.output or _default_output(args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
