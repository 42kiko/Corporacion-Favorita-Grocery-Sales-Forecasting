"""
Unit tests for src/data_loader.py
Covers YAML loading, file detection, preprocessing, and sampling logic.
"""

from __future__ import annotations
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from src.data_loader import DataLoader, load_yaml, ensure_dir

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def create_dummy_csv(tmp_path: Path) -> Path:
    """Create small dummy CSV with mixed types (like Favorita train)."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2017-01-01", "2017-01-02", "2017-01-03"],
            "store_nbr": [1, 1, 2],
            "item_nbr": [101, 102, 103],
            "unit_sales": [10.5, 20.0, 30.1],
            "onpromotion": ["True", "False", None],
        }
    )
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_load_yaml_reads_existing(tmp_path: Path) -> None:
    cfg = {"source": {"local_dir": "data", "use_parquet_first": True}}
    yaml_path = tmp_path / "active.yaml"
    yaml_path.write_text("source:\n  local_dir: data\n  use_parquet_first: true\n")
    data = load_yaml(yaml_path)
    assert data["source"]["use_parquet_first"] is True


def test_ensure_dir_creates_folder(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested"
    assert not new_dir.exists()
    ensure_dir(new_dir)
    assert new_dir.exists()


def test_detect_and_preprocess_to_parquet(tmp_path: Path) -> None:
    """Verify CSV → Parquet conversion runs and keeps schema consistent."""
    # Prepare dummy file
    csv_path = create_dummy_csv(tmp_path)

    # Dummy config YAML
    cfg_yaml = tmp_path / "active.yaml"
    cfg_yaml.write_text(
        """
source:
  local_dir: {tmp}
  use_parquet_first: true
  drive_link: ""
preprocess:
  enabled: true
  compression: "zstd"
  deduplicate: true
  coerce_dates: true
  normalize_booleans: true
  chunksize_rows: 2
region:
  column: ""
  value: ""
sample:
  enabled: false
""".format(tmp=tmp_path)
    )

    # Run DataLoader
    loader = DataLoader(config_path=str(cfg_yaml))
    loader.preprocess_csv_to_parquet()

    # Check if Parquet exists
    parquet_file = tmp_path / "train.parquet"
    assert parquet_file.exists(), "Expected Parquet file after preprocessing"

    # Validate schema consistency
    parquet_table = pq.read_table(parquet_file)
    cols = [f.name for f in parquet_table.schema]
    assert "onpromotion" in cols
    onpromotion_type = str(parquet_table.schema.field("onpromotion").type)
    assert "bool" in onpromotion_type, f"Expected bool type, got {onpromotion_type}"


def test_load_train_data_pipeline(tmp_path: Path) -> None:
    """Ensure full pipeline runs without crashing."""
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
    assert "onpromotion" in df.columns