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
import time


# ============================================================
# --- Utility Functions
# ============================================================

def log(msg: str) -> None:
    """Simple console logger for notebooks & CLI."""
    print(msg)


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
        self.use_parquet_first = self.config["source"].get("use_parquet_first", True)

    # --------------------------------------------------------
    def detect_files(self) -> Dict[str, Path]:
        """Return available CSV or Parquet files in local_dir."""
        files: Dict[str, Path] = {}
        for ext in ["parquet", "csv"]:
            for f in self.local_dir.glob(f"*.{ext}"):
                files[f.stem] = f
        return files

    # --------------------------------------------------------
    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load dataset and merge split parts if needed."""
        ensure_dir(self.local_dir)

        part_files = sorted(self.local_dir.glob(f"{name}_part*.parquet"))
        single_file = self.local_dir / f"{name}.parquet"

        if part_files:
            log(f"📦 Found {len(part_files)} parts for {name}. Merging…")
            df = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)

        elif single_file.exists():
            log(f"📦 Loading {single_file.name} …")
            df = pd.read_parquet(single_file)

        else:
            csv_file = self.local_dir / f"{name}.csv"
            if csv_file.exists():
                log(f"📄 Loading {csv_file.name} (CSV fallback)…")
                df = pd.read_csv(csv_file, low_memory=False)
            else:
                raise FileNotFoundError(f"No dataset '{name}' found in {self.local_dir}")

        log(f"✅ Loaded {len(df):,} rows × {len(df.columns)} cols.")
        return df

    # --------------------------------------------------------
    def preprocess_csv_to_parquet(self, files: Optional[list[str]] = None) -> None:
        """Convert CSVs into Parquet (split or partitioned)."""

        preprocess_cfg = self.config["preprocess"]
        if not preprocess_cfg.get("enabled", False):
            log("ℹ️ Preprocessing disabled in config.")
            return

        compression = preprocess_cfg.get("compression", "zstd")
        chunksize = preprocess_cfg.get("chunksize_rows", 1_000_000)
        part_size = preprocess_cfg.get("part_size_rows", 5_000_000)
        partition_cols = preprocess_cfg.get("partition_by", [])
        dedup_global = preprocess_cfg.get("deduplicate_global", True)

        ensure_dir(self.local_dir)

        candidates = list(self.local_dir.glob("*.csv"))
        if files:
            wanted = {p.lower() for p in files}
            candidates = [p for p in candidates if p.name.lower() in wanted]

        if not candidates:
            log("⚠️ No CSV files found.")
            return

        for csv_file in candidates:
            base_name = csv_file.stem
            log(f"\n➡️ Processing {csv_file.name} …")
            t0 = time.time()

            reader = pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)
            buffer: list[pd.DataFrame] = []
            total_rows = 0
            part_idx = 1

            for chunk in reader:
                total_rows += len(chunk)

                # --- cleaning ---
                if preprocess_cfg.get("deduplicate", True):
                    key_cols = [c for c in ["id", "date", "store_nbr"] if c in chunk.columns]
                    if key_cols:
                        chunk = chunk.drop_duplicates(subset=key_cols)

                if preprocess_cfg.get("coerce_dates", True) and "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

                if "onpromotion" in chunk.columns:
                    chunk["onpromotion"] = (
                        chunk["onpromotion"].astype(str).str.lower().str.strip()
                        .replace({"true": True, "false": False, "nan": None, "": None})
                        .astype("boolean")
                    )

                # add partition columns
                if partition_cols and "date" in chunk.columns:
                    if "year" in partition_cols:
                        chunk["year"] = chunk["date"].dt.year
                    if "month" in partition_cols:
                        chunk["month"] = chunk["date"].dt.month
                    if "day" in partition_cols:
                        chunk["day"] = chunk["date"].dt.day

                buffer.append(chunk)

                # --- write when buffer full ---
                if sum(len(b) for b in buffer) >= part_size:
                    part_df = pd.concat(buffer, ignore_index=True)
                    buffer.clear()

                    table = pa.Table.from_pandas(part_df, preserve_index=False)

                    if partition_cols:
                        pq.write_to_dataset(
                            table,
                            root_path=self.local_dir / base_name,
                            partition_cols=partition_cols,
                            compression=compression,
                        )
                        log(f"🧩 Wrote partitioned chunk ({len(part_df):,} rows)")
                    else:
                        out_path = self.local_dir / f"{base_name}_part{part_idx}.parquet"
                        pq.write_table(table, out_path, compression=compression)
                        log(f"🧩 Wrote {out_path.name} ({len(part_df):,} rows)")
                        part_idx += 1

            # leftovers
            if buffer:
                part_df = pd.concat(buffer, ignore_index=True)
                table = pa.Table.from_pandas(part_df, preserve_index=False)

                if partition_cols:
                    pq.write_to_dataset(
                        table,
                        root_path=self.local_dir / base_name,
                        partition_cols=partition_cols,
                        compression=compression,
                    )
                    log(f"🧩 Wrote final partitioned chunk ({len(part_df):,} rows)")
                else:
                    out_path = self.local_dir / f"{base_name}_part{part_idx}.parquet"
                    pq.write_table(table, out_path, compression=compression)
                    log(f"🧩 Wrote {out_path.name} ({len(part_df):,} rows)")

            dt = time.time() - t0
            log(f"✅ Done: {base_name} processed in {dt:,.1f}s")

            # optional global dedup
            if dedup_global:
                part_files = sorted(self.local_dir.glob(f"{base_name}_part*.parquet"))
                if part_files:
                    df_full = pd.concat((pd.read_parquet(p) for p in part_files), ignore_index=True)
                    before = len(df_full)

                    # recognize valid key columns for this dataset
                    key_cols = [c for c in ["id", "date", "store_nbr"] if c in df_full.columns]

                    if key_cols:
                        # only deduplicate when valid key columns exist
                        df_full = df_full.drop_duplicates(subset=key_cols)
                        after = len(df_full)
                        log(f"🧹 Global dedup: {before:,} → {after:,}")
                    else:
                        # no dedup possible → skip safely
                        log(f"ℹ️ No global dedup for '{base_name}' (missing key columns).")
                        after = before

                    # rewrite parquet parts (split into same chunk size)
                    for p in part_files:
                        p.unlink(missing_ok=True)

                    idx = 1
                    for i in range(0, len(df_full), part_size):
                        chunk = df_full.iloc[i:i + part_size]
                        out_path = self.local_dir / f"{base_name}_part{idx}.parquet"
                        pq.write_table(
                            pa.Table.from_pandas(chunk, preserve_index=False),
                            out_path,
                            compression=compression
                        )
                        idx += 1

                    log("💾 Rewrote deduplicated parts")
            log(f"🎯 Finalized {base_name}")

    # --------------------------------------------------------
    def apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        s_cfg = self.config["sample"]
        if not s_cfg.get("enabled", False):
            return df

        log("🔍 Applying sampling …")
        if s_cfg["mode"] == "frac":
            df = df.sample(frac=s_cfg["frac"], random_state=42)
        else:
            df = df.sample(n=min(len(df), s_cfg["n_rows"]), random_state=42)

        log(f"✅ Sampled {len(df):,} rows.")
        return df

    # --------------------------------------------------------
    def apply_region_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        r_cfg = self.config["region"]
        col, val = r_cfg.get("column"), r_cfg.get("value")

        if not col or not val:
            return df

        if col not in df.columns:
            log(f"⚠️ Column '{col}' not found — skipping region filter")
            return df

        log(f"🌎 Filtering by {col} = '{val}'")
        df = df[df[col] == val]
        log(f"✅ Filtered to {len(df):,} rows.")
        return df

    # --------------------------------------------------------
    def load_train_data(self) -> pd.DataFrame:
        df = self.load_dataset("train")
        df = self.apply_region_filter(df)
        df = self.apply_sampling(df)
        return df