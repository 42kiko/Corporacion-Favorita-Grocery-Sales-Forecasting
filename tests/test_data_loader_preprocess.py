"""
Unit tests for DataLoader.preprocess_csv_to_parquet()
Ensures correct CSV → Parquet split conversion, deduplication,
boolean normalization, and overall schema integrity.
"""

from __future__ import annotations
import pytest
import pandas as pd
import yaml
from pathlib import Path
from src.data_loader import DataLoader, ensure_dir

# ------------------------------------------------------------
# --- Fixtures
# ------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temp directory with one sample train.csv file."""
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
    """Write a temporary YAML config file for preprocessing."""
    cfg = {
        "source": {
            "local_dir": str(tmp_data_dir),
            "use_parquet_first": True,
            "drive_link": ""
        },
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


# ------------------------------------------------------------
# --- Tests
# ------------------------------------------------------------

def test_preprocess_creates_split_parquet(tmp_config: Path, tmp_data_dir: Path) -> None:
    """
    Ensure CSV → Parquet split conversion:
    - creates multiple part files,
    - removes duplicate (id, date, store_nbr) combinations,
    - keeps valid boolean normalization for 'onpromotion'.
    """
    loader = DataLoader(config_path=tmp_config)
    loader.preprocess_csv_to_parquet()

    # Expect at least one split file
    part_files = sorted(tmp_data_dir.glob("train_part*.parquet"))
    assert part_files, "Expected at least one split Parquet part file"

    # Merge parts for inspection
    df = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)

    # --- Structural checks ---
    assert set(df.columns) == {"id", "date", "store_nbr", "unit_sales", "onpromotion"}, \
        "Unexpected columns in Parquet output"

    # --- Deduplication check ---
    key_cols = ["id", "date", "store_nbr"]
    duplicates = df.duplicated(subset=key_cols).sum()
    assert duplicates == 0, "Duplicate (id, date, store_nbr) combos should be removed"

    # --- Boolean normalization check ---
    assert df["onpromotion"].dtype.name in ("boolean", "bool"), "'onpromotion' should be boolean"
    assert df["onpromotion"].notna().any(), "There should be valid True/False values"

    print(f"✅ {len(part_files)} Parquet parts validated, {len(df):,} rows total")


def test_preprocess_handles_no_csvs_gracefully(tmp_path: Path) -> None:
    """Ensure preprocess_csv_to_parquet() does not crash when no CSVs exist."""
    cfg = {
        "source": {"local_dir": str(tmp_path), "use_parquet_first": True, "drive_link": ""},
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

    loader = DataLoader(config_path=cfg_path)
    # Should run without raising exceptions
    loader.preprocess_csv_to_parquet()