from __future__ import annotations

import sys
from types import SimpleNamespace

from agrimanager.adapter.utils import AgriGenerationsLogger


class FakeTable:
    def __init__(self, columns, data=None):
        self.columns = list(columns)
        self.data = list(data or [])

    def add_data(self, *values):
        self.data.append(list(values))


def test_agri_generations_logger_records_validation_metadata(monkeypatch):
    logged = []
    fake_wandb = SimpleNamespace(
        Table=FakeTable,
        run=object(),
        log=lambda payload, step=None: logged.append((payload, step)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = AgriGenerationsLogger()
    logger.log(
        ["wandb"],
        [
            {
                "validation_set": "drought",
                "validation_axis": "weather_regime",
                "validation_axis_value": "drought",
                "weather_regime": "drought",
                "crop_regime": "",
                "scenario_id": "scenario-1",
                "crop_name": "chickpea",
                "trajectory": "prompt and answer",
                "reward": 1.25,
                "num_steps": 3,
            }
        ],
        step=7,
    )

    table = logged[0][0]["val-table/agri_generations"]
    assert "validation_set" in table.columns
    assert "validation_axis_value" in table.columns
    assert "scenario_id" in table.columns
    row = dict(zip(table.columns, table.data[0]))
    assert row["validation_set"] == "drought"
    assert row["validation_axis"] == "weather_regime"
    assert row["validation_axis_value"] == "drought"
    assert row["weather_regime"] == "drought"
    assert row["scenario_id"] == "scenario-1"
    assert row["crop_name"] == "chickpea"
