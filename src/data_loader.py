"""
src/data_loader.py

Handles all data loading, preprocessing, and sampling logic.
Uses configs/data/active.yaml to decide where and how to load data.
"""

from __future__ import annotations
import os
from os import PathLike
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
import time


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


# ============================================================
# --- Core Data Loader
# ============================================================

class DataLoader:
    """Main data access and preprocessing logic."""

    def __init__(self, config_path: Union[str, Path, PathLike[str]] = "configs/data/active.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        self.local_dir = Path(self.config["source"]["local_dir"])
        self.use_parquet_first = self.config["source"]["use_parquet_first"]

    # --------------------------------------------------------
    # 1. File detection
    # --------------------------------------------------------
    def detect_files(self) -> Dict[str, Path]:
        """Return available CSV or Parquet files in local_dir."""
        files: Dict[str, Path] = {}
        for ext in ["parquet", "csv"]:
            for f in self.local_dir.glob(f"*.{ext}"):
                files[f.stem] = f
        return files

    # --------------------------------------------------------
    # 2. Load dataset (handles split files)
    # --------------------------------------------------------
    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        Load a dataset, automatically merging split Parquet files (e.g. train_part1.parquet … train_partN.parquet)
        while applying region filter and sampling from YAML configuration.
        """
        ensure_dir(self.local_dir)
        part_files = sorted(self.local_dir.glob(f"{name}_part*.parquet"))
        single_file = self.local_dir / f"{name}.parquet"

        if part_files:
            st.info(f"📦 Found {len(part_files)} parts for {name}. Merging …")
            dfs = [pd.read_parquet(p) for p in part_files]
            df = pd.concat(dfs, ignore_index=True)
        elif single_file.exists():
            st.info(f"📦 Loading {single_file.name} …")
            df = pd.read_parquet(single_file)
        else:
            csv_file = self.local_dir / f"{name}.csv"
            if csv_file.exists():
                st.info(f"📄 Loading {csv_file.name} (CSV fallback) …")
                df = pd.read_csv(csv_file, low_memory=False)
            else:
                raise FileNotFoundError(f"No dataset '{name}' found in {self.local_dir}")

        st.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns.")
        return df

    # --------------------------------------------------------
    # 3. Preprocess CSV → Parquet
    # --------------------------------------------------------
    def preprocess_csv_to_parquet(self, files: Optional[list[str]] = None) -> None:
        """
        Convert large CSVs to Parquet safely (streaming + schema-stable)
        and perform a final global deduplication if enabled in config.
        """
        preprocess_cfg = self.config["preprocess"]
        if not preprocess_cfg["enabled"]:
            st.info("ℹ️ Preprocessing disabled in config.")
            return

        compression = preprocess_cfg["compression"]
        chunksize = preprocess_cfg["chunksize_rows"]

        ensure_dir(self.local_dir)
        candidates = list(self.local_dir.glob("*.csv"))
        if files:
            wanted = {f.lower() for f in files}
            candidates = [p for p in candidates if p.name.lower() in wanted]

        if not candidates:
            st.warning("⚠️ No CSVs found.")
            return

        for csv_file in candidates:
            parquet_file = self.local_dir / f"{csv_file.stem}.parquet"
            st.write(f"➡️ {csv_file.name} → {parquet_file.name}")
            t0 = time.time()

            reader = pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)
            writer: Optional[pq.ParquetWriter] = None
            total_rows = 0
            chunk_idx = 0

            for chunk in reader:
                chunk_idx += 1
                total_rows += len(chunk)

                # --- Normalize 'onpromotion' BEFORE deduplication ---
                if "onpromotion" in chunk.columns:
                    chunk["onpromotion"] = (
                        chunk["onpromotion"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace({"true": True, "false": False, "nan": None, "": None})
                        .fillna("false")  # ensure no NA left for dedup
                        .replace({"false": False, "true": True})
                        .astype(bool)
                    )

                # --- Coerce dates before deduplication ---
                if preprocess_cfg["coerce_dates"] and "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

                # --- Now safe deduplication ---
                if preprocess_cfg["deduplicate"]:
                    key_cols = [c for c in ["id", "date", "store_nbr"] if c in chunk.columns]
                    if key_cols:
                        chunk = chunk.drop_duplicates(subset=key_cols, keep="first")

                # --- Normalize numeric consistency ---
                for col in chunk.columns:
                    if pd.api.types.is_float_dtype(chunk[col]):
                        chunk[col] = chunk[col].astype("float64")

                # → Arrow Table
                table = pa.Table.from_pandas(chunk, preserve_index=False)

                # Maintain schema stability
                if writer is None:
                    writer = pq.ParquetWriter(parquet_file, table.schema, compression=compression)
                else:
                    table = table.cast(writer.schema)

                writer.write_table(table)

                if chunk_idx % 10 == 0:
                    st.write(f"   · processed {total_rows:,} rows so far...")

            if writer:
                writer.close()

            # --- Global deduplication after all chunks ---
            if preprocess_cfg["deduplicate"]:
                st.info("🧹 Performing global deduplication (after merge)...")
                df_full = pd.read_parquet(parquet_file)
                before = len(df_full)
                key_cols = [c for c in ["id", "date", "store_nbr"] if c in df_full.columns]
                if key_cols:
                    df_full = df_full.drop_duplicates(subset=key_cols, keep="first")
                after = len(df_full)
                df_full.to_parquet(parquet_file, compression=compression, index=False)
                st.success(f"✅ Global deduplication: {before:,} → {after:,} rows.")


    # --------------------------------------------------------
    # 4. Sampling logic
    # --------------------------------------------------------
    def apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a sampled subset if enabled."""
        s_cfg = self.config["sample"]
        if not s_cfg["enabled"]:
            return df

        st.info("🔍 Applying sampling …")
        if s_cfg["mode"] == "frac":
            df = df.sample(frac=s_cfg["frac"], random_state=42)
        else:
            df = df.sample(n=min(len(df), s_cfg["n_rows"]), random_state=42)

        st.success(f"✅ Sampled {len(df):,} rows.")
        return df

    # --------------------------------------------------------
    # 5. Region filtering (from YAML)
    # --------------------------------------------------------
    def apply_region_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset by region.column/value if configured."""
        r_cfg = self.config["region"]
        col, val = r_cfg.get("column"), r_cfg.get("value")

        if not col or not val:
            return df
        if col not in df.columns:
            st.warning(f"⚠️ Column '{col}' not found in dataset.")
            return df

        st.info(f"🌎 Filtering by {col} = '{val}' …")
        df = df[df[col] == val]
        st.success(f"✅ Filtered to {len(df):,} rows.")
        return df

    # --------------------------------------------------------
    # 6. High-level convenience method
    # --------------------------------------------------------
    def load_train_data(self) -> pd.DataFrame:
        """Load, merge, region-filter and sample train data."""
        df = self.load_dataset("train")
        df = self.apply_region_filter(df)
        df = self.apply_sampling(df)
        return df


# ============================================================
# --- Streamlit integration helper
# ============================================================

def render_data_section_ui() -> None:
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