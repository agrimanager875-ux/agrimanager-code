"""Shared crop-trait schema definitions and path resolution helpers."""

from __future__ import annotations

from pathlib import Path


DEFAULT_CROP_TRAIT_SCHEMA = "traits_v1_23d"
POLICY_CROP_TRAIT_SCHEMA = "traits_v1_6d"
STRATEGY_SELECTED_CROP_TRAIT_SCHEMA = "traits_strategy_selected_v1"
RICE_VARIETY_TRAIT_SCHEMA = "rice_variety_traits_v1"
SUPPORTED_CROP_TRAIT_SCHEMAS = (
    DEFAULT_CROP_TRAIT_SCHEMA,
    POLICY_CROP_TRAIT_SCHEMA,
    STRATEGY_SELECTED_CROP_TRAIT_SCHEMA,
    RICE_VARIETY_TRAIT_SCHEMA,
)


def normalize_crop_trait_schema(trait_schema: str | None) -> str:
    """Return a validated trait schema name."""
    schema = str(trait_schema or DEFAULT_CROP_TRAIT_SCHEMA).strip()
    if schema not in SUPPORTED_CROP_TRAIT_SCHEMAS:
        raise ValueError(
            f"Unsupported crop trait schema {schema!r}. "
            f"Supported schemas: {list(SUPPORTED_CROP_TRAIT_SCHEMAS)}"
        )
    return schema


def resolve_crop_trait_schema_dir(base_dir: str | Path, trait_schema: str | None) -> Path:
    """Resolve the directory that contains schema-specific crop trait artifacts.

    The current repository keeps legacy 23D cards directly under ``base_dir`` for
    backward compatibility, while schema-aware cards may also live under
    ``base_dir/<schema>/``. For the default schema, prefer the subdirectory when
    present and otherwise fall back to the legacy flat layout.
    """

    base_path = Path(base_dir).resolve()
    schema = normalize_crop_trait_schema(trait_schema)
    if base_path.name == schema and base_path.is_dir():
        return base_path
    schema_dir = (base_path / schema).resolve()
    if schema_dir.is_dir():
        return schema_dir
    if schema == DEFAULT_CROP_TRAIT_SCHEMA:
        return base_path
    raise FileNotFoundError(
        f"Crop trait schema directory not found for schema {schema!r}: {schema_dir}"
    )


def resolve_crop_trait_artifact_path(
    base_dir: str | Path,
    crop_name: str,
    extension: str,
    trait_schema: str | None,
) -> Path:
    """Resolve a schema-aware crop trait artifact path."""

    ext = extension if extension.startswith(".") else f".{extension}"
    schema_dir = resolve_crop_trait_schema_dir(base_dir, trait_schema)
    return schema_dir / f"{crop_name}{ext}"


def crop_variety_trait_key(crop_name: str, crop_variety: str | None) -> str:
    """Return the trait artifact key for a crop variety."""
    crop = str(crop_name or "").strip()
    variety = str(crop_variety or "").strip()
    if not crop or not variety:
        return crop
    return f"{crop}__{variety}"
