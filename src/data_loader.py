"""
src/data_loader.py

Handles all data loading, preprocessing, and sampling logic.
Uses configs/data/active.yaml to decide where and how to load data.
"""

from __future__ import annotations
import os
from os import PathLike
import yaml
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as pv
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import gdown


# ============================================================
# --- Utility Functions
# ============================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Safely read YAML into dict."""
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def download_from_drive(folder_link: str, dest: Path) -> None:
    """Download all CSVs from Google Drive folder using gdown."""
    if not folder_link:
        st.warning("No Google Drive link provided.")
        return
    ensure_dir(dest)
    st.info(f"📥 Downloading data from Google Drive to {dest} ...")
    gdown.download_folder(url=folder_link, output=str(dest), quiet=False, use_cookies=False)
    st.success("✅ Download complete!")


# ============================================================
# --- Core Data Loader
# ============================================================

class DataLoader:
    def __init__(self, config_path: str | Path | PathLike[str] = "configs/data/active.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        self.local_dir = Path(self.config["source"]["local_dir"])
        self.use_parquet_first = self.config["source"]["use_parquet_first"]
        self.drive_link = self.config["source"].get("drive_link", "")

    # --------------------------------------------------------
    # 1. File detection & loading
    # --------------------------------------------------------
    def detect_files(self) -> Dict[str, Path]:
        """Return available CSV or Parquet files in local_dir."""
        data_files = {}
        for ext in ["parquet", "csv"]:
            for f in self.local_dir.glob(f"*.{ext}"):
                key = f.stem
                data_files[key] = f
        return data_files

    # --------------------------------------------------------
    # 2. Load a single dataset (auto Parquet/CSV fallback)
    # --------------------------------------------------------
    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load one dataset (e.g., train, stores) with automatic Parquet/CSV handling."""
        files = self.detect_files()

        # Attempt auto-download if no data found
        if not files:
            st.warning("⚠️ No data files found locally. Attempting Drive download...")
            download_from_drive(self.drive_link, self.local_dir)
            files = self.detect_files()

        # Auto-create Parquet if none exist but CSVs are present
        parquet_exists = any(f.suffix == ".parquet" for f in files.values())
        csv_exists = any(f.suffix == ".csv" for f in files.values())

        if self.use_parquet_first and not parquet_exists and csv_exists:
            st.info("🧩 No Parquet found — running one-time preprocessing...")
            self.preprocess_csv_to_parquet()
            files = self.detect_files()

        # Load according to priority
        if self.use_parquet_first and f"{name}" in files and files[name].suffix == ".parquet":
            st.info(f"📦 Loading {name}.parquet ...")
            return pd.read_parquet(files[name])
        elif f"{name}" in files:
            st.info(f"📄 Loading {name}.csv ...")
            return pd.read_csv(files[name], low_memory=False)
        else:
            raise FileNotFoundError(f"Dataset '{name}' not found in {self.local_dir}")

    # --------------------------------------------------------
    # 3. Preprocess: CSV → Parquet
    # --------------------------------------------------------
    def preprocess_csv_to_parquet(self) -> None:
        """Convert all CSVs to Parquet according to config."""
        preprocess_cfg = self.config["preprocess"]
        if not preprocess_cfg["enabled"]:
            st.info("ℹ️ Preprocessing disabled in config.")
            return

        compression = preprocess_cfg["compression"]
        partition_by = preprocess_cfg["partition_by"]
        chunksize = preprocess_cfg["chunksize_rows"]

        st.info("🧩 Starting CSV → Parquet conversion...")
        ensure_dir(self.local_dir)

        for csv_file in self.local_dir.glob("*.csv"):
            parquet_file = self.local_dir / f"{csv_file.stem}.parquet"
            st.write(f"➡️ Converting {csv_file.name} → {parquet_file.name}")

            # Stream in chunks for large CSVs
            reader = pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)
            table_parts = []
            for chunk in reader:
                if preprocess_cfg["deduplicate"]:
                    chunk = chunk.drop_duplicates()
                if preprocess_cfg["coerce_dates"] and "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                if preprocess_cfg["normalize_booleans"]:
                    for col in chunk.select_dtypes(include=["bool"]).columns:
                        chunk[col] = chunk[col].astype(int)
                table_parts.append(pa.Table.from_pandas(chunk))

            table = pa.concat_tables(table_parts)
            pq.write_table(table, parquet_file, compression=compression)
        st.success("✅ All CSV files converted to Parquet!")

    # --------------------------------------------------------
    # 4. Sampling logic
    # --------------------------------------------------------
    def apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a sampled subset if enabled."""
        s_cfg = self.config["sample"]
        if not s_cfg["enabled"]:
            return df

        st.info("🔍 Applying sampling...")
        if s_cfg["mode"] == "frac":
            df = df.sample(frac=s_cfg["frac"], random_state=42)
        else:
            df = df.sample(n=min(len(df), s_cfg["n_rows"]), random_state=42)
        st.success(f"✅ Sampled {len(df):,} rows.")
        return df

    # --------------------------------------------------------
    # 5. Region filtering
    # --------------------------------------------------------
    def apply_region_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by region.column and region.value if provided."""
        r_cfg = self.config["region"]
        col, val = r_cfg["column"], r_cfg["value"]
        if not col or not val:
            return df
        if col not in df.columns:
            st.warning(f"⚠️ Column '{col}' not found in dataset.")
            return df
        st.info(f"🌎 Filtering by {col} = '{val}' ...")
        return df[df[col] == val]

    # --------------------------------------------------------
    # 6. Master load function
    # --------------------------------------------------------
    def load_train_data(self) -> pd.DataFrame:
        """Load, preprocess (if needed), sample, and filter train data."""
        df = self.load_dataset("train")
        df = self.apply_region_filter(df)
        df = self.apply_sampling(df)
        return df


# ============================================================
# --- Streamlit integration helper
# ============================================================

def render_data_section_ui():
    """Streamlit UI to trigger data-related actions."""
    st.header("📂 Data Management")

    loader = DataLoader()

    col1, col2 = st.columns(2)
    if col1.button("⚙️ Preprocess CSV → Parquet"):
        loader.preprocess_csv_to_parquet()
    if col2.button("📊 Load Train Sample"):
        df = loader.load_train_data()
        st.dataframe(df.head(50))
        st.success(f"Loaded dataset with {len(df):,} rows.")