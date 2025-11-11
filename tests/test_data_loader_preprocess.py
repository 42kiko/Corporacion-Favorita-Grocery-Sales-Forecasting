"""
tests/test_data_loader_preprocess.py

Verifies that DataLoader.preprocess_csv_to_parquet() converts CSVs correctly,
handles deduplication, normalizes booleans, and writes valid Parquet output.
"""

from pathlib import Path
import pandas as pd
import yaml
import pytest
from src.data_loader import DataLoader, ensure_dir


# ------------------------------------------------------------
# --- Fixtures
# ------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with one sample CSV file."""
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
    """Write a temporary YAML config file."""
    cfg = {
        "source": {
            "local_dir": str(tmp_data_dir),
            "use_parquet_first": True
        },
        "sample": {
            "enabled": False,
            "mode": "frac",
            "frac": 0.1,
            "n_rows": 2
        },
        "region": {
            "column": None,
            "value": None
        },
        "preprocess": {
            "enabled": True,
            "compression": "snappy",
            "chunksize_rows": 1000,
            "deduplicate": True,
            "coerce_dates": True,
            "normalize_booleans": True
        }
    }

    cfg_path = tmp_path / "active.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


# ------------------------------------------------------------
# --- Tests
# ------------------------------------------------------------

def test_preprocess_creates_parquet(tmp_config: Path, tmp_data_dir: Path) -> None:
    """
    Ensure CSV → Parquet conversion:
    - creates file,
    - removes true duplicates (same id/date/store_nbr),
    - keeps valid variations in 'onpromotion'.
    """
    loader = DataLoader(config_path=tmp_config)
    loader.preprocess_csv_to_parquet()

    parquet_file = tmp_data_dir / "train.parquet"
    assert parquet_file.exists(), "Parquet file should be created"

    df = pd.read_parquet(parquet_file)

    # --- Basic structure ---
    assert set(df.columns) == {"id", "date", "store_nbr", "unit_sales", "onpromotion"}

    # --- Deduplication check ---
    key_cols = ["id", "date", "store_nbr"]
    duplicates = df.duplicated(subset=key_cols).sum()
    assert duplicates == 0, "Duplicate (id, date, store_nbr) combos should be removed"
    assert len(df) == 3, "There should be exactly 3 unique rows after deduplication"

    # --- Boolean normalization check ---
    assert "onpromotion" in df.columns
    assert df["onpromotion"].dtype.name in ("boolean", "bool")

    print("✅ Final deduplicated dataset:")
    print(df)