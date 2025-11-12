"""
Unit tests for src/data_loader.py
Covers YAML loading, directory creation, preprocessing (CSV→Parquet split),
and loading pipeline integration.
"""

from __future__ import annotations
import pytest
import pandas as pd
import yaml
from pathlib import Path
from src.data_loader import DataLoader, load_yaml, ensure_dir

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data dir with one small train.csv file."""
    data_dir = tmp_path / "data"
    ensure_dir(data_dir)

    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
        "store_nbr": [1, 1, 1, 2],
        "unit_sales": [10.5, 11.0, 11.0, 15.2],
        "onpromotion": ["True", "False", "true", None],
    })
    df.to_csv(data_dir / "train.csv", index=False)
    return data_dir


@pytest.fixture
def tmp_config(tmp_path: Path, tmp_data_dir: Path) -> Path:
    """Write minimal YAML config for DataLoader."""
    cfg = {
        "source": {"local_dir": str(tmp_data_dir), "use_parquet_first": True, "drive_link": ""},
        "preprocess": {
            "enabled": True,
            "compression": "snappy",
            "deduplicate": True,
            "coerce_dates": True,
            "normalize_booleans": True,
            "chunksize_rows": 1000
        },
        "region": {"column": "", "value": ""},
        "sample": {"enabled": False}
    }

    cfg_path = tmp_path / "active.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def create_dummy_csv(tmp_path: Path) -> Path:
    """Create small dummy CSV similar to Favorita format."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "date": ["2017-01-01", "2017-01-02", "2017-01-03"],
        "store_nbr": [1, 1, 2],
        "item_nbr": [101, 102, 103],
        "unit_sales": [10.5, 20.0, 30.1],
        "onpromotion": ["True", "False", None],
    })
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_load_yaml_reads_existing(tmp_path: Path) -> None:
    yaml_path = tmp_path / "active.yaml"
    yaml_path.write_text("source:\n  local_dir: data\n  use_parquet_first: true\n")
    data = load_yaml(yaml_path)
    assert data["source"]["use_parquet_first"] is True


def test_ensure_dir_creates_folder(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested"
    assert not new_dir.exists()
    ensure_dir(new_dir)
    assert new_dir.exists()


def test_preprocess_creates_parquet(tmp_config: Path, tmp_data_dir: Path) -> None:
    """
    Ensure CSV → Parquet split conversion:
    - creates multiple part files,
    - removes duplicates,
    - normalizes booleans correctly.
    """
    loader = DataLoader(config_path=tmp_config)
    loader.preprocess_csv_to_parquet()

    part_files = sorted(tmp_data_dir.glob("train_part*.parquet"))
    assert part_files, "Expected at least one Parquet part file"

    df = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)

    # Deduplication check
    key_cols = ["id", "date", "store_nbr"]
    assert df.duplicated(subset=key_cols).sum() == 0, "Duplicate (id, date, store_nbr) should be removed"
    assert "onpromotion" in df.columns

    # Boolean normalization check
    assert df["onpromotion"].dtype.name in ("boolean", "bool")
    assert df["onpromotion"].notna().any(), "Expected at least one valid boolean value"


def test_load_train_data_pipeline(tmp_path: Path) -> None:
    """Ensure full pipeline runs and loads DataFrame correctly."""
    # Create dummy CSV + config
    create_dummy_csv(tmp_path)
    cfg_yaml = tmp_path / "active.yaml"
    cfg_yaml.write_text(
        f"""
source:
  local_dir: "{tmp_path}"
  use_parquet_first: false
  drive_link: ""
preprocess:
  enabled: false
  compression: "zstd"
  deduplicate: false
  coerce_dates: false
  normalize_booleans: false
  chunksize_rows: 1000
region:
  column: ""
  value: ""
sample:
  enabled: false
"""
    )

    loader = DataLoader(config_path=str(cfg_yaml))
    df = loader.load_train_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"id", "date", "unit_sales"}.issubset(df.columns)