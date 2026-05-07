from __future__ import annotations

from pathlib import Path

import pytest

from agrimanager.env.wofost_gym.crop_trait_schemas import (
    DEFAULT_CROP_TRAIT_SCHEMA,
    POLICY_CROP_TRAIT_SCHEMA,
    resolve_crop_trait_schema_dir,
)
from agrimanager.env.wofost_gym.crop_traits_observation import CropTraitEncoder
from agrimanager.env.wofost_gym.env import WOFOSTEnv
from agrimanager.env.wofost_gym.env_config import (
    DEFAULT_CROP_TRAITS_DIR,
    DEFAULT_WOFOST_GYM_PATH,
    REPO_ROOT,
    WOFOSTEnvConfig,
)


@pytest.mark.parametrize(
    ("trait_schema", "expected_dim", "expected_feature"),
    [
        (
            DEFAULT_CROP_TRAIT_SCHEMA,
            23,
            "core_facts.phenology.TSUM_total_Cd",
        ),
        (
            POLICY_CROP_TRAIT_SCHEMA,
            6,
            "core_facts.decision_axes.water_priority",
        ),
    ],
)
def test_crop_trait_encoder_supports_both_schemas(
    trait_schema: str,
    expected_dim: int,
    expected_feature: str,
) -> None:
    encoder = CropTraitEncoder(trait_schema=trait_schema)

    assert encoder.dim == expected_dim
    assert expected_feature in encoder.feature_names


def test_schema_dir_resolution_prefers_schema_subdirs() -> None:
    base_dir = (REPO_ROOT / DEFAULT_CROP_TRAITS_DIR).resolve()

    assert resolve_crop_trait_schema_dir(base_dir, DEFAULT_CROP_TRAIT_SCHEMA).is_dir()
    assert resolve_crop_trait_schema_dir(base_dir, POLICY_CROP_TRAIT_SCHEMA).is_dir()


@pytest.mark.parametrize(
    ("trait_schema", "expected_text"),
    [
        (DEFAULT_CROP_TRAIT_SCHEMA, "Nutrient capacity proxies"),
        (POLICY_CROP_TRAIT_SCHEMA, "Decision Profile"),
    ],
)
def test_wofost_env_loads_schema_specific_prompt_traits(
    trait_schema: str,
    expected_text: str,
) -> None:
    config = WOFOSTEnvConfig(
        env_id="lnpkw-v0",
        agro_file="wheat_agro.yaml",
        wofost_gym_path=DEFAULT_WOFOST_GYM_PATH,
        llm_mode=True,
        include_crop_traits=True,
        trait_schema=trait_schema,
        turn_num=1,
        seed=0,
    )
    env = WOFOSTEnv(config)

    try:
        prompt, _ = env.reset()
        assert "<crop traits>" in prompt
        assert expected_text in prompt
    finally:
        env.close()
