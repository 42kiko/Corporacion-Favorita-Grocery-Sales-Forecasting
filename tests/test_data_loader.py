"""
Unit tests for src/data_loader.py
Covers YAML loading, directory creation, sampling, region filter, and dataset detection.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import pytest

# --- Fix import path so pytest finds src ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loader import DataLoader, load_yaml, ensure_dir


# ============================================================
# --- Fixtures
# ============================================================

@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create a minimal valid active.yaml config file."""
    config_dir = tmp_path / "configs" / "data"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = """\
source:
  local_dir: "{data_dir}"
  use_parquet_first: false
  drive_link: ""
preprocess:
  enabled: true
  compression: "zstd"
  chunksize_rows: 1000
  deduplicate: true
  coerce_dates: true
  normalize_booleans: true
region:
  column: ""
  value: ""
sample:
  enabled: false
  mode: "frac"
  frac: 0.1
  n_rows: 1000
"""
    # Substitute temp dir for local_dir
    yaml_text = yaml_content.format(data_dir=str(tmp_path / "data"))
    yaml_path = config_dir / "active.yaml"
    yaml_path.write_text(yaml_text)
    return yaml_path


@pytest.fixture
def mock_csv_data(tmp_path: Path) -> pd.DataFrame:
    """Create a mock CSV dataset for testing."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "store_nbr": [1, 1, 2, 2, 3],
        "unit_sales": [10, 20, 30, 40, 50],
        "city": ["A", "A", "B", "B", "C"]
    })
    data_dir = tmp_path / "data"
    ensure_dir(data_dir)
    df.to_csv(data_dir / "train.csv", index=False)
    return df


# ============================================================
# --- Tests
# ============================================================

def test_load_yaml_valid(temp_config: Path) -> None:
    """Ensure load_yaml correctly reads YAML."""
    config = load_yaml(temp_config)
    assert isinstance(config, dict)
    assert "source" in config
    assert config["source"]["use_parquet_first"] is False


def test_ensure_dir_creates_nested(tmp_path: Path) -> None:
    """Ensure nested directories are created without error."""
    new_dir = tmp_path / "nested" / "folder"
    ensure_dir(new_dir)
    assert new_dir.exists() and new_dir.is_dir()


def test_detect_files_finds_csv(temp_config: Path, mock_csv_data: pd.DataFrame) -> None:
    """DataLoader should detect CSV files inside local_dir."""
    loader = DataLoader(config_path=temp_config)
    files = loader.detect_files()
    assert "train" in files
    assert files["train"].suffix == ".csv"


def test_load_dataset_reads_csv(temp_config: Path, mock_csv_data: pd.DataFrame) -> None:
    """DataLoader.load_dataset should return a DataFrame from CSV."""
    loader = DataLoader(config_path=temp_config)
    df = loader.load_dataset("train")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "unit_sales" in df.columns


def test_apply_region_filter_filters_correctly(temp_config: Path, mock_csv_data: pd.DataFrame) -> None:
    """apply_region_filter should filter dataset by region settings."""
    loader = DataLoader(config_path=temp_config)
    # Modify config in-memory for region filter
    loader.config["region"]["column"] = "city"
    loader.config["region"]["value"] = "A"
    filtered = loader.apply_region_filter(mock_csv_data)
    assert len(filtered) == 2
    assert all(filtered["city"] == "A")


def test_apply_sampling_frac(temp_config: Path, mock_csv_data: pd.DataFrame) -> None:
    """apply_sampling should return a smaller sample when enabled."""
    loader = DataLoader(config_path=temp_config)
    loader.config["sample"]["enabled"] = True
    loader.config["sample"]["mode"] = "frac"
    loader.config["sample"]["frac"] = 0.4
    sampled = loader.apply_sampling(mock_csv_data)
    assert len(sampled) < len(mock_csv_data)
    assert isinstance(sampled, pd.DataFrame)