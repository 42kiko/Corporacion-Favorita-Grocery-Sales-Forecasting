# tests/test_ui_data_config.py
import tempfile
from pathlib import Path
import yaml
import pytest

from src import ui_data_config


def test_load_yaml_valid(tmp_path: Path):
    """Ensure load_yaml reads valid YAML into a dict."""
    test_file = tmp_path / "test.yaml"
    test_data = {"source": {"local_dir": "data"}}
    test_file.write_text(yaml.safe_dump(test_data))

    result = ui_data_config.load_yaml(test_file)
    assert isinstance(result, dict)
    assert result["source"]["local_dir"] == "data"


def test_load_yaml_missing_file(caplog):
    """Missing file should return empty dict and log error."""
    result = ui_data_config.load_yaml(Path("nonexistent.yaml"))
    assert result == {}


def test_save_yaml_roundtrip(tmp_path: Path):
    """Data saved and loaded should remain identical."""
    path = tmp_path / "out.yaml"
    data = {"a": 1, "b": {"c": 2}}
    ui_data_config.save_yaml(path, data)

    reloaded = yaml.safe_load(path.read_text())
    assert reloaded == data


def test_schema_structure():
    """Validate schema.yaml structure for required top-level sections."""
    schema_path = Path("configs/data/schema.yaml")
    assert schema_path.exists(), "schema.yaml missing"

    schema = yaml.safe_load(schema_path.read_text())
    for section in ["source", "preprocess", "window", "region", "sample", "test_data", "validation"]:
        assert section in schema, f"Missing section: {section}"