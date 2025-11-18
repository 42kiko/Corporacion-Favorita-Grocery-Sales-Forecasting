# ============================================
# src/plots/overview.py
# Typed, reusable Plotly utilities for:
# - EDA Overview Notebook
# - Streamlit interactive dashboard
# ============================================

from __future__ import annotations
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Helper: Ensure a datetime column exists in the DataFrame
# ============================================================
def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Ensure the column `col` is converted to a pandas datetime dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a date column.
    col : str
        Column name to convert (default: 'date').

    Returns
    -------
    pd.DataFrame
        The same DataFrame, with `df[col]` converted to datetime.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ============================================================
# 1) Store Distribution (Count of stores appearing in the data)
# ============================================================
def plot_store_distribution(train: pd.DataFrame) -> go.Figure:
    """
    Plot a simple distribution of store IDs based on row counts.

    Parameters
    ----------
    train : pd.DataFrame
        Dataset containing at least `store_nbr`.

    Returns
    -------
    go.Figure
        Plotly bar figure.
    """
    store_counts = train["store_nbr"].value_counts().sort_index()

    fig = px.bar(
        x=store_counts.index,
        y=store_counts.values,
        labels={"x": "Store Number", "y": "Count"},
        title="Store Distribution (Aggregated)",
    )
    return fig


# ============================================================
# 2) Top Stores by Average Sales
# ============================================================
def plot_top_stores_avg_sales(train: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Plot the stores with the highest average unit sales.

    Parameters
    ----------
    train : pd.DataFrame
        Dataset containing store_nbr and unit_sales.
    top_n : int
        Number of stores to display.

    Returns
    -------
    go.Figure
        Plotly bar chart.
    """
    avg_sales = (
        train.groupby("store_nbr")["unit_sales"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = px.bar(
        avg_sales,
        x="store_nbr",
        y="unit_sales",
        title=f"Top {top_n} Stores by Average Sales",
        labels={"store_nbr": "Store", "unit_sales": "Avg Unit Sales"},
    )
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ============================================================
# 3) Top Items by Frequency (+ Metadata)
# ============================================================
def plot_top_items(
    train: pd.DataFrame,
    items_meta: pd.DataFrame,
    top_n: int = 30,
) -> go.Figure:
    """
    Plot the Top-N most frequent items based on row counts and
    enrich them with metadata from the items dataset.

    Guarantees:
    - All item labels appear on the left side.
    - Plot height scales automatically.
    - Category order is fixed to preserve the data order.

    Parameters
    ----------
    train : pd.DataFrame
        Must contain 'item_nbr'.
    items_meta : pd.DataFrame
        Must contain columns: 'item_nbr', 'family', 'class', 'perishable'.
    top_n : int
        Number of items to display.

    Returns
    -------
    go.Figure
        Horizontal bar chart with metadata hover details.
    """

    # ---- SAFETY GUARDS ----
    if not isinstance(train, pd.DataFrame):
        raise TypeError(f"'train' must be a DataFrame, got {type(train)}")

    if not isinstance(items_meta, pd.DataFrame):
        raise TypeError(
            f"'items_meta' must be a DataFrame, but got {type(items_meta)}"
        )

    required_cols = {"item_nbr", "family", "class", "perishable"}
    if not required_cols.issubset(items_meta.columns):
        raise ValueError(
            f"'items_meta' must contain {required_cols}, "
            f"but only has {set(items_meta.columns)}"
        )

    # ---- 1) Count frequencies ----
    item_counts = (
        train["item_nbr"]
        .value_counts()
        .rename_axis("item_nbr")
        .reset_index(name="count")
    )

    top_items = item_counts.head(top_n).copy()

    # ---- 2) Join metadata ----
    items_meta = items_meta[list(required_cols)].drop_duplicates()
    top_items = top_items.merge(items_meta, on="item_nbr", how="left")

    # ---- 3) Build readable labels ----
    def make_label(row):
        if pd.notna(row["family"]):
            return f"{row['family']} (#{int(row['item_nbr'])})"
        return f"Item #{int(row['item_nbr'])}"

    top_items["label"] = top_items.apply(make_label, axis=1)

    # ---- 4) Sort for horizontal bar chart ----
    top_items = top_items.sort_values("count", ascending=True)
    label_order = top_items["label"].tolist()

    # ---- 5) Plot ----
    fig = px.bar(
        top_items,
        x="count",
        y="label",
        orientation="h",
        title=f"Top {top_n} Most Frequent Items (with Families)",
        labels={"count": "Number of Sales Records", "label": "Item"},
        hover_data=["item_nbr", "family", "class", "perishable"],
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=label_order,
        automargin=True,
    )

    fig.update_layout(
        height=max(700, top_n * 30),
        margin=dict(l=300, r=40, t=70, b=40),
    )

    return fig


# ============================================================
# 4) Distribution of sales rows per item
# ============================================================
def plot_unit_sales_distribution(train: pd.DataFrame, bins: int = 80) -> go.Figure:
    """
    Histogram of how many rows each item appears in.

    Parameters
    ----------
    train : pd.DataFrame
        Must contain 'item_nbr'.
    bins : int
        Histogram bins.

    Returns
    -------
    go.Figure
    """
    item_counts = (
        train["item_nbr"]
        .value_counts()
        .rename_axis("item_nbr")
        .reset_index(name="n_rows")
    )

    fig = px.histogram(
        item_counts,
        x="n_rows",
        nbins=bins,
        title="Distribution of Sales Records per Item",
        labels={"n_rows": "Number of Rows (Sales Records)"},
    )
    fig.update_layout(bargap=0.05)
    return fig


# ============================================================
# 5) Total unit sales over time
# ============================================================
def plot_total_sales_over_time(
    train: pd.DataFrame,
    date_range: Optional[Tuple[str, str]] = None,
) -> go.Figure:
    """
    Time-series of total daily sales.

    Parameters
    ----------
    train : pd.DataFrame
        Must contain 'date' and 'unit_sales'.
    date_range : (str, str), optional
        Filter range: ('YYYY-MM-DD', 'YYYY-MM-DD').

    Returns
    -------
    go.Figure
    """
    df = _ensure_datetime(train.copy())

    if date_range:
        start, end = date_range
        df = df[(df["date"] >= start) & (df["date"] <= end)]

    ts = (
        df.groupby("date", as_index=False)["unit_sales"]
        .sum()
        .sort_values("date")
    )

    fig = px.line(
        ts,
        x="date",
        y="unit_sales",
        title="Total Sales Over Time",
        labels={"unit_sales": "Total Unit Sales"},
    )
    return fig


# ============================================================
# 6) Average sales by weekday
# ============================================================
def plot_avg_sales_by_weekday(train: pd.DataFrame) -> go.Figure:
    """
    Show weekly seasonality by grouping average daily sales per weekday.

    Returns
    -------
    go.Figure
    """
    train["day_of_week"] = pd.to_datetime(train["date"]).dt.day_name()

    dow = (
        train.groupby("day_of_week")["unit_sales"]
        .mean()
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        .reset_index()
    )

    fig = px.bar(
        dow,
        x="day_of_week",
        y="unit_sales",
        title="Average Sales by Day of Week",
        template="plotly_dark",
    )
    return fig


# ============================================================
# 7) Promotions vs Sales
# ============================================================
def plot_promo_vs_sales(train: pd.DataFrame) -> go.Figure:
    """
    Compare average unit sales between promoted vs. non-promoted rows.

    Returns
    -------
    go.Figure
    """
    promo_df = train.groupby("onpromotion")["unit_sales"].mean().reset_index()

    fig = px.bar(
        promo_df,
        x="onpromotion",
        y="unit_sales",
        title="Average Sales vs Promotion Status",
        template="plotly_dark",
    )
    return fig