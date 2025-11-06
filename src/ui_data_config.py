import yaml
import streamlit as st
from pathlib import Path
from typing import Any, Dict

# Utility paths
CONFIG_DIR = Path("configs/data")
SCHEMA_PATH = CONFIG_DIR / "schema.yaml"
ACTIVE_PATH = CONFIG_DIR / "active.yaml"

def load_yaml(path: Path) -> Dict[str, Any]:
    """Safely load YAML file into a Python dict."""
    if not path.exists():
        st.error(f"Config file missing: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write dict back to YAML with safe formatting."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def render_field(key: str, meta: Dict[str, Any], value: Any):
    """Render one UI control based on type information from schema."""
    t = meta.get("type", "str")
    help_text = meta.get("help", "")
    if t == "bool":
        return st.checkbox(key, value=bool(value), help=help_text)
    elif t == "int":
        return st.number_input(
            key,
            min_value=meta.get("min"),
            max_value=meta.get("max"),
            step=meta.get("step", 1),
            value=int(value),
            help=help_text,
        )
    elif t == "float":
        return st.number_input(
            key,
            min_value=meta.get("min"),
            max_value=meta.get("max"),
            step=meta.get("step", 0.01),
            value=float(value),
            format="%.4f",
            help=help_text,
        )
    elif t == "str":
        choices = meta.get("choices")
        if choices:
            return st.selectbox(key, choices, index=choices.index(value) if value in choices else 0, help=help_text)
        else:
            return st.text_input(key, value=value or "", help=help_text)
    elif t in {"str_list", "int_list"}:
        raw = ", ".join(str(x) for x in (value or []))
        return st.text_input(key, raw, help=f"{help_text} (comma-separated list)")
    else:
        st.warning(f"Unknown type {t} for field {key}")
        return value


def render_data_config_ui(stores_meta=None) -> Dict[str, Any]:
    """Render full data config section dynamically from schema and active YAML."""
    st.header("📂 Data Configuration")

    schema = load_yaml(SCHEMA_PATH)
    active = load_yaml(ACTIVE_PATH)

    if not schema or not active:
        st.error("Missing schema or active config.")
        return active

    updated = {}

    # --- Source section
    st.subheader("Data Source")
    updated["source"] = {}
    for k, meta in schema["source"].items():
        val = active.get("source", {}).get(k, meta.get("default"))
        updated["source"][k] = render_field(f"source.{k}", meta, val)

    # --- Preprocessing section
    st.subheader("Preprocessing")
    updated["preprocess"] = {}
    for k, meta in schema["preprocess"].items():
        val = active.get("preprocess", {}).get(k, meta.get("default"))
        updated["preprocess"][k] = render_field(f"preprocess.{k}", meta, val)

    # --- Window section
    st.subheader("Window")
    updated["window"] = {}
    for k, meta in schema["window"].items():
        val = active.get("window", {}).get(k, meta.get("default"))
        updated["window"][k] = render_field(f"window.{k}", meta, val)

    # --- Region section (dynamic choices)
    st.subheader("Region")
    updated["region"] = {}
    region_schema = schema["region"]

    # Populate dynamic choices if meta provided (stores_meta from stores.parquet)
    col_choices = region_schema["column"].get("choices", [])
    if stores_meta is not None:
        possible_cols = [c for c in stores_meta.columns if c not in ["store_nbr", "store_name"]]
        region_schema["column"]["choices"] = sorted(list(set(col_choices + possible_cols)))

    for k, meta in region_schema.items():
        val = active.get("region", {}).get(k, meta.get("default"))
        updated["region"][k] = render_field(f"region.{k}", meta, val)

    # --- Sampling section
    st.subheader("Sampling")
    updated["sample"] = {}
    for k, meta in schema["sample"].items():
        val = active.get("sample", {}).get(k, meta.get("default"))
        updated["sample"][k] = render_field(f"sample.{k}", meta, val)

    # --- Test data section
    st.subheader("Test Data")
    updated["test_data"] = {}
    for k, meta in schema["test_data"].items():
        val = active.get("test_data", {}).get(k, meta.get("default"))
        updated["test_data"][k] = render_field(f"test_data.{k}", meta, val)

    # --- Validation section (read-only for now)
    with st.expander("Validation settings (read-only)"):
        for k, meta in schema["validation"].items():
            val = active.get("validation", {}).get(k, meta.get("default"))
            st.text(f"{k}: {val}")

    # Save button
    if st.button("💾 Save Data Config"):
        save_yaml(ACTIVE_PATH, updated)
        st.success("Configuration saved successfully.")

    return updated