"""
tests/test_data_loader_split_merge.py

Ensures that DataLoader correctly merges split train_part*.parquet files
and applies region/sampling logic based on YAML config.
"""

from pathlib import Path
import pandas as pd
import pytest
import yaml
from src.data_loader import DataLoader, ensure_dir

# ------------------------------------------------------------
# --- Fixtures
# ------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a fake local_dir with multiple split parquet parts."""
    data_dir = tmp_path / "data"
    ensure_dir(data_dir)

    # Create 3 small fake Parquet parts
    for i in range(3):
        df = pd.DataFrame({
            "id": [i * 10 + j for j in range(3)],
            "city": ["Quito", "Quito", "Guayaquil"],
            "sales": [100 + i * 5 + j for j in range(3)]
        })
        df.to_parquet(data_dir / f"train_part{i+1}.parquet")

    return data_dir


@pytest.fixture
def tmp_config(tmp_path: Path, tmp_data_dir: Path) -> Path:
    """Generate a minimal active.yaml for DataLoader."""
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
            "column": "city",
            "value": "Quito"
        },
        "preprocess": {
            "enabled": False,
            "compression": "snappy",
            "chunksize_rows": 500000,
            "deduplicate": True,
            "coerce_dates": True
        }
    }

    cfg_path = tmp_path / "active.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path


# ------------------------------------------------------------
# --- Tests
# ------------------------------------------------------------

def test_merge_split_parquets(tmp_config: Path) -> None:
    """Merging split Parquet files produces combined DataFrame."""
    loader = DataLoader(config_path=tmp_config)
    df = loader.load_dataset("train")

    # Should combine 3 parts × 3 rows = 9 rows total
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9
    assert "sales" in df.columns


def test_region_filter_applied(tmp_config: Path) -> None:
    """Region filter from YAML correctly filters to 'Quito' rows."""
    loader = DataLoader(config_path=tmp_config)
    df_full = loader.load_dataset("train")
    df_filtered = loader.apply_region_filter(df_full)

    # Expect only rows where city == "Quito"
    assert set(df_filtered["city"].unique()) == {"Quito"}
    assert len(df_filtered) < len(df_full)


def test_sampling_disabled_by_default(tmp_config: Path) -> None:
    """Sampling does not alter row count when disabled."""
    loader = DataLoader(config_path=tmp_config)
    df = loader.load_dataset("train")
    df_sampled = loader.apply_sampling(df)

    # Sampling disabled → identical row count
    assert len(df_sampled) == len(df)