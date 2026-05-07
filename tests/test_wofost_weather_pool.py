from __future__ import annotations

import os
import tarfile
from pathlib import Path

from agrimanager.env.wofost_gym.weather_pool import (
    DEFAULT_POOL_REPO_ID,
    METEO_CACHE_ARCHIVE_NAME,
    ensure_pool,
    find_pool_meteo_cache_dir,
)


def test_configure_pcse_weather_cache_enables_cache_only_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agrimanager.env.wofost_gym.env import _configure_pcse_weather_cache

    cache_dir = tmp_path / "meteo_cache"
    monkeypatch.delenv("PCSE_METEO_CACHE_DIR", raising=False)
    monkeypatch.delenv("PCSE_NASAPOWER_NO_RETRIEVE", raising=False)

    _configure_pcse_weather_cache(str(cache_dir))

    assert cache_dir.is_dir()
    assert Path(os.environ["PCSE_METEO_CACHE_DIR"]) == cache_dir
    assert os.environ["PCSE_NASAPOWER_NO_RETRIEVE"] == "1"


def test_configure_pcse_weather_cache_respects_explicit_retrieve_setting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agrimanager.env.wofost_gym.env import _configure_pcse_weather_cache

    monkeypatch.setenv("PCSE_NASAPOWER_NO_RETRIEVE", "0")

    _configure_pcse_weather_cache(str(tmp_path / "meteo_cache"))

    assert os.environ["PCSE_NASAPOWER_NO_RETRIEVE"] == "0"


def test_find_pool_meteo_cache_dir_extracts_archive(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    archive_path = pool_dir / METEO_CACHE_ARCHIVE_NAME
    source_root = tmp_path / "source"
    source_cache_dir = source_root / "meteo_cache"
    source_cache_dir.mkdir(parents=True)
    payload_path = source_cache_dir / "example.cache"
    payload_path.write_bytes(b"example-weather-cache")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_cache_dir, arcname="meteo_cache")

    extracted_dir = find_pool_meteo_cache_dir(pool_dir)

    assert extracted_dir == (pool_dir / "meteo_cache").resolve()
    assert (extracted_dir / "example.cache").read_bytes() == b"example-weather-cache"


def test_non_default_hf_pool_uses_repo_specific_cache(tmp_path: Path, monkeypatch) -> None:
    from agrimanager.env.wofost_gym import weather_pool

    calls: list[tuple[str, Path, str]] = []
    monkeypatch.setattr(weather_pool, "_REPO_ROOT", tmp_path)

    def fake_download(repo_id: str, local_dir: str | Path, revision: str = "main") -> Path:
        local = Path(local_dir)
        calls.append((repo_id, local, revision))
        split_dir = local / "val_drought"
        split_dir.mkdir(parents=True)
        (split_dir / "chickpea.parquet").write_bytes(b"placeholder")
        return local

    monkeypatch.setattr(weather_pool, "download_pool", fake_download)

    pool_dir = ensure_pool("agrimanager/wofost-weather-regime-pool", revision="rev-a")

    assert calls
    assert calls[0][0] == "agrimanager/wofost-weather-regime-pool"
    assert calls[0][1].name == "agrimanager__wofost-weather-regime-pool"
    assert calls[0][2] == "rev-a"
    assert pool_dir == calls[0][1]

    calls.clear()
    pool_dir_again = ensure_pool("agrimanager/wofost-weather-regime-pool", revision="rev-b")

    assert pool_dir_again == pool_dir
    assert calls == []


def test_default_hf_pool_refreshes_legacy_no_test_cache(tmp_path: Path, monkeypatch) -> None:
    from agrimanager.env.wofost_gym import weather_pool

    legacy_dir = tmp_path / "legacy-default-pool"
    (legacy_dir / "train").mkdir(parents=True)
    (legacy_dir / "val").mkdir()
    (legacy_dir / "train" / "wheat.parquet").write_bytes(b"placeholder")
    (legacy_dir / "val" / "wheat.parquet").write_bytes(b"placeholder")
    calls: list[tuple[str, Path, str]] = []

    def fake_download(repo_id: str, local_dir: str | Path, revision: str = "main") -> Path:
        local = Path(local_dir)
        calls.append((repo_id, local, revision))
        return local

    monkeypatch.setattr(weather_pool, "download_pool", fake_download)

    pool_dir = ensure_pool(DEFAULT_POOL_REPO_ID, revision="rev-a", local_dir=legacy_dir)

    assert pool_dir == legacy_dir
    assert calls == [(DEFAULT_POOL_REPO_ID, legacy_dir, "rev-a")]
