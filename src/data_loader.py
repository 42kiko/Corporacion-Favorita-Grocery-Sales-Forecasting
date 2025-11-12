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
        Convert large CSVs into Parquet parts or partitioned datasets.
        - Supports partitioning by ['year','month'] or other columns
        - Ensures schema consistency, deduplication, and boolean/date normalization
        - Writes to multiple Parquet parts or partition folders
        """
        preprocess_cfg = self.config["preprocess"]
        if not preprocess_cfg.get("enabled", False):
            st.info("ℹ️ Preprocessing disabled in config.")
            return

        compression = preprocess_cfg.get("compression", "zstd")
        chunksize = preprocess_cfg.get("chunksize_rows", 1_000_000)
        part_size = preprocess_cfg.get("part_size_rows", 5_000_000)
        partition_cols = preprocess_cfg.get("partition_by", [])
        dedup_global = preprocess_cfg.get("deduplicate_global", True)

        ensure_dir(self.local_dir)
        candidates = list(self.local_dir.glob("*.csv"))
        if files:
            wanted = {f.lower() for f in files}
            candidates = [p for p in candidates if p.name.lower() in wanted]

        if not candidates:
            st.warning("⚠️ No CSV files found.")
            return

        for csv_file in candidates:
            base_name = csv_file.stem
            st.write(f"➡️ Processing {csv_file.name} …")
            t0 = time.time()

            reader = pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)
            buffer: list[pd.DataFrame] = []
            total_rows = 0
            part_idx = 1

            for chunk in reader:
                total_rows += len(chunk)

                # --- Step 1: Light cleaning ---
                if preprocess_cfg.get("deduplicate", True):
                    key_cols = [c for c in ["id", "date", "store_nbr"] if c in chunk.columns]
                    if key_cols:
                        chunk = chunk.drop_duplicates(subset=key_cols, keep="first")

                if preprocess_cfg.get("coerce_dates", True) and "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

                if "onpromotion" in chunk.columns:
                    chunk["onpromotion"] = (
                        chunk["onpromotion"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace({"true": True, "false": False, "nan": None, "": None})
                        .astype("boolean")
                    )

                # --- Step 2: Add partition columns dynamically ---
                if partition_cols and "date" in chunk.columns:
                    if "year" in partition_cols and "year" not in chunk.columns:
                        chunk["year"] = chunk["date"].dt.year
                    if "month" in partition_cols and "month" not in chunk.columns:
                        chunk["month"] = chunk["date"].dt.month
                    if "day" in partition_cols and "day" not in chunk.columns:
                        chunk["day"] = chunk["date"].dt.day

                buffer.append(chunk)

                # --- Step 3: Write Parquet when buffer is full ---
                if sum(len(b) for b in buffer) >= part_size:
                    part_df = pd.concat(buffer, ignore_index=True)
                    buffer.clear()

                    if partition_cols:
                        # Partitioned write
                        table = pa.Table.from_pandas(part_df, preserve_index=False)
                        pq.write_to_dataset(
                            table,
                            root_path=self.local_dir / base_name,
                            partition_cols=partition_cols,
                            compression=compression
                        )
                        st.write(f"🧩 Wrote partitioned chunk ({len(part_df):,} rows)")
                    else:
                        # Classic part file
                        parquet_file = self.local_dir / f"{base_name}_part{part_idx}.parquet"
                        table = pa.Table.from_pandas(part_df, preserve_index=False)
                        pq.write_table(table, parquet_file, compression=compression)
                        st.write(f"🧩 Wrote {parquet_file.name} ({len(part_df):,} rows)")
                        part_idx += 1

            # --- Step 4: Write leftovers ---
            if buffer:
                part_df = pd.concat(buffer, ignore_index=True)
                if partition_cols:
                    table = pa.Table.from_pandas(part_df, preserve_index=False)
                    pq.write_to_dataset(
                        table,
                        root_path=self.local_dir / base_name,
                        partition_cols=partition_cols,
                        compression=compression
                    )
                    st.write(f"🧩 Wrote final partitioned chunk ({len(part_df):,} rows)")
                else:
                    parquet_file = self.local_dir / f"{base_name}_part{part_idx}.parquet"
                    table = pa.Table.from_pandas(part_df, preserve_index=False)
                    pq.write_table(table, parquet_file, compression=compression)
                    st.write(f"🧩 Wrote {parquet_file.name} ({len(part_df):,} rows)")

            dt = time.time() - t0
            st.success(f"✅ Done: {base_name} processed in {dt:,.1f}s")

            # --- Step 5: Optional global deduplication ---
            if dedup_global:
                all_parts = sorted(self.local_dir.glob(f"{base_name}_part*.parquet"))
                if all_parts:
                    df_full = pd.concat([pd.read_parquet(p) for p in all_parts], ignore_index=True)
                    before = len(df_full)
                    key_cols = [c for c in ["id", "date", "store_nbr"] if c in df_full.columns]
                    if key_cols:
                        df_full = df_full.drop_duplicates(subset=key_cols, keep="first")
                        after = len(df_full)
                        st.info(f"🧹 Global deduplication: {before:,} → {after:,} rows")
                        # rewrite
                        for old in all_parts:
                            old.unlink(missing_ok=True)
                        for i in range(0, len(df_full), part_size):
                            part_df = df_full.iloc[i:i+part_size]
                            out_path = self.local_dir / f"{base_name}_part{i//part_size + 1}.parquet"
                            table = pa.Table.from_pandas(part_df, preserve_index=False)
                            pq.write_table(table, out_path, compression=compression)
                            st.write(f"💾 Rewrote {out_path.name} ({len(part_df):,} rows)")

            st.success(f"🎯 Finalized {base_name}")

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