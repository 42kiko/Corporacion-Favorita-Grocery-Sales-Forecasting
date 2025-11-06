"""
tests/test_data_loader.py

Unit tests for src/data_loader.py
"""

from pathlib import Path
import pandas as pd
import yaml
import pytest

from src.data_loader import DataLoader


# ---------------------------------------------------------------------
# Helper: Create a fake active.yaml in a temporary folder
# ---------------------------------------------------------------------
@pytest.fixture
def temp_active_yaml(tmp_path: Path):
    config_dir = tmp_path / "configs" / "data"
    config_dir.mkdir(parents=True)

    active_cfg = {
        "source": {
            "local_dir": str(tmp_path / "data"),
            "drive_link": "",
            "use_parquet_first": True,
        },
        "preprocess": {
            "enabled": True,
            "partition_by": [],
            "compression": "zstd",
            "deduplicate": True,
            "coerce_dates": True,
            "normalize_booleans": True,
            "chunksize_rows": 1000,
        },
        "region": {"column": "state", "value": "Azuay"},
        "sample": {
            "enabled": True,
            "mode": "frac",
            "frac": 0.5,
            "n_rows": 10000,
            "stratify_by": "",
        },
        "window": {"mode": "months", "months": 12, "horizon_days": 14},
        "test_data": {"prefer_local": True, "allow_manual_upload": True, "align_on": []},
        "validation": {"require_core_files": ["train.csv"]},
    }

    with open(config_dir / "active.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(active_cfg, f)

    return config_dir / "active.yaml"


# ---------------------------------------------------------------------
# Helper: Create minimal CSV file
# ---------------------------------------------------------------------
@pytest.fixture
def sample_csv(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "store_nbr": [1, 2, 3, 4],
            "sales": [10, 20, 30, 40],
            "state": ["Azuay", "Pichincha", "Azuay", "Cotopaxi"],
        }
    )
    csv_path = data_dir / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_load_yaml_and_detect_files(temp_active_yaml, sample_csv):
    """Ensure config and data file detection works."""
    loader = DataLoader(config_path=temp_active_yaml)
    files = loader.detect_files()
    assert "train" in files, "train.csv should be detected"


def test_load_dataset_csv(temp_active_yaml, sample_csv):
    """Load CSV file correctly into DataFrame."""
    loader = DataLoader(config_path=temp_active_yaml)
    df = loader.load_dataset("train")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "sales" in df.columns


def test_preprocess_csv_to_parquet_creates_file(temp_active_yaml, sample_csv):
    """Convert CSV → Parquet and verify output exists."""
    loader = DataLoader(config_path=temp_active_yaml)
    loader.preprocess_csv_to_parquet()

    parquet_path = Path(loader.local_dir) / "train.parquet"
    assert parquet_path.exists(), "Parquet file should have been created"
    df = pd.read_parquet(parquet_path)
    assert not df.empty


def test_apply_region_filter(temp_active_yaml, sample_csv):
    """Filter only rows where state == 'Azuay'."""
    loader = DataLoader(config_path=temp_active_yaml)
    df = loader.load_dataset("train")
    df_filtered = loader.apply_region_filter(df)
    assert all(df_filtered["state"] == "Azuay"), "Filtered rows should match region"


def test_apply_sampling(temp_active_yaml, sample_csv):
    """Ensure sampling reduces row count."""
    loader = DataLoader(config_path=temp_active_yaml)
    df = loader.load_dataset("train")
    sampled = loader.apply_sampling(df)
    assert len(sampled) <= len(df), "Sampled DataFrame should be smaller"