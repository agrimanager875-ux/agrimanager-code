"""Crop-trait helpers for cycles_gym prompt injection.

This package stores short, simulator-backed crop cards used to teach the LLM
what each candidate crop means in the current rotation action space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


CROP_TRAITS_DIR = Path(__file__).resolve().parent


def available_crop_traits() -> list[str]:
    """Return the set of crop names with maintained prompt trait cards."""
    return sorted(path.stem for path in CROP_TRAITS_DIR.glob("*.txt"))


def load_crop_traits(crop_name: str) -> str:
    """Load the maintained trait card for one crop."""
    traits_path = CROP_TRAITS_DIR / f"{crop_name}.txt"
    if not traits_path.exists():
        raise FileNotFoundError(
            f"Crop traits not found for cycles_gym crop '{crop_name}': {traits_path}"
        )
    return traits_path.read_text(encoding="utf-8").strip()


def build_crop_traits_text(rotation_crops: Iterable[str]) -> str:
    """Build one prompt block covering the current rotation crops only."""
    sections = [load_crop_traits(crop_name) for crop_name in rotation_crops]
    return "\n\n".join(section for section in sections if section.strip())


__all__ = [
    "CROP_TRAITS_DIR",
    "available_crop_traits",
    "load_crop_traits",
    "build_crop_traits_text",
]
