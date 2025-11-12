# tests/test_features_config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


# ---------- Helpers ----------

def _is_list_of_ints(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, int) for i in x)

def _is_list_of_strs(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)

def _read_yaml(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"Missing YAML file: {path}"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"Invalid YAML structure in {path}"
    return data

# ---------- B) schema.yaml + active.yaml ----------

@pytest.mark.skipif(not Path("configs/features/schema.yaml").exists(), reason="schema.yaml not present")
def test_features_schema_yaml_has_expected_sections_and_types():
    """
    Validates the richer feature schema (if present) based auf der zuvor besprochenen Struktur.
    Wir prüfen nur die Existenz & grobe Typen, nicht jeden Grenzwert.
    """
    path = Path("configs/features/schema.yaml")
    schema = _read_yaml(path)

    # Expected top-level groups
    expected_groups = [
        "calendar",
        "lags",
        "rolling",
        "expanding",
        "holidays",
        "exogenous",
        "categoricals",
        "interactions",
        "target",
    ]
    for g in expected_groups:
        assert g in schema, f"Missing group '{g}' in {path}"

    # Spot-check some fields (type hints presence)
    assert "enabled" in schema["calendar"]
    assert "values" in schema["lags"]
    assert "windows" in schema["rolling"] and "stats" in schema["rolling"]
    assert "transform" in schema["target"]


@pytest.mark.skipif(not Path("configs/features/active.yaml").exists(), reason="active.yaml not present")
def test_features_active_yaml_aligns_with_schema_if_both_exist():
    """
    Wenn schema.yaml & active.yaml vorhanden sind: prüfe, dass active grob die gleichen Gruppen enthält
    und Werte sinnvolle Typen haben.
    """
    schema_path = Path("configs/features/schema.yaml")
    active_path = Path("configs/features/active.yaml")
    schema = _read_yaml(schema_path)
    active = _read_yaml(active_path)

    # Gleiche Gruppen wie im Schema (Subset-Check)
    for group in schema.keys():
        assert group in active, f"Active config missing group '{group}'"

    # Typ-Checks (Spot-Checks, nicht exhaustiv)
    # calendar
    assert isinstance(active["calendar"]["enabled"], bool)
    assert _is_list_of_strs(active["calendar"]["fields"])

    # lags
    assert isinstance(active["lags"]["enabled"], bool)
    assert _is_list_of_ints(active["lags"]["values"])

    # rolling
    assert isinstance(active["rolling"]["enabled"], bool)
    assert _is_list_of_ints(active["rolling"]["windows"])
    assert _is_list_of_strs(active["rolling"]["stats"])

    # holidays
    assert isinstance(active["holidays"]["enabled"], bool)
    assert isinstance(active["holidays"]["proximity_days"], int)

    # target
    assert "transform" in active["target"]
    assert active["target"]["transform"] in {"none", "log1p"}