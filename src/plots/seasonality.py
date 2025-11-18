"""
Seasonality plotting utilities.

This module provides reusable Plotly figures for:
- building a clean daily time series
- visualizing daily sales and rolling averages
- weekly and monthly seasonality
- month x weekday heatmaps
- STL decomposition of the daily series

All functions are designed to be used both in notebooks and Streamlit.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def build_daily_series(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
) -> pd.DataFrame:
    """
    Aggregate a raw transaction level frame to a daily time series.

    Parameters
    ----------
    df:
        Input DataFrame with at least [date_col, target_col].
    date_col:
        Name of the date column.
    target_col:
        Name of the numeric target column to aggregate.

    Returns
    -------
    pd.DataFrame
        Frame with columns [date_col, target_col] aggregated by day.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame")

    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame")

    ts = df[[date_col, target_col]].copy()

    if not pd.api.types.is_datetime64_any_dtype(ts[date_col]):
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")

    ts = (
        ts.groupby(date_col, as_index=False)[target_col]
        .sum()
        .sort_values(date_col)
    )

    return ts


# ---------------------------------------------------------------------------
# Daily sales and rolling averages
# ---------------------------------------------------------------------------

def plot_daily_sales(
    ts_daily: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
) -> go.Figure:
    """
    Line plot of daily total sales.
    """
    fig = px.line(
        ts_daily,
        x=date_col,
        y=target_col,
        title="Daily Total Unit Sales",
        labels={date_col: "Date", target_col: "Total Unit Sales"},
    )
    return fig


def plot_rolling_averages(
    ts_daily: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
    windows: Iterable[int] = (7, 30, 90),
) -> go.Figure:
    """
    Plot daily series together with multiple rolling means.

    Parameters
    ----------
    ts_daily:
        Daily aggregated frame.
    date_col:
        Date column name.
    target_col:
        Target column name.
    windows:
        Iterable of window sizes in days.
    """
    df = ts_daily[[date_col, target_col]].copy()
    windows = list(windows)

    for w in windows:
        col_name = f"ma_{w}"
        df[col_name] = df[target_col].rolling(window=w, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[target_col],
            mode="lines",
            name="Daily",
            opacity=0.3,
        )
    )

    for w in windows:
        col_name = f"ma_{w}"
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col_name],
                mode="lines",
                name=f"{w} day MA",
            )
        )

    fig.update_layout(
        title="Daily Sales with Rolling Averages",
        xaxis_title="Date",
        yaxis_title="Unit Sales",
    )
    return fig


# ---------------------------------------------------------------------------
# Weekly and monthly seasonality
# ---------------------------------------------------------------------------

_WEEKDAY_MAP: Dict[int, str] = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

_MONTH_LABELS: Dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Internal helper that returns a copy with a proper datetime column."""
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def plot_weekday_seasonality(
    ts_daily: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
) -> go.Figure:
    """
    Plot average daily sales per weekday.
    """
    df = _ensure_datetime(ts_daily[[date_col, target_col]], date_col)
    df["weekday"] = df[date_col].dt.dayofweek
    df["weekday_name"] = df["weekday"].map(_WEEKDAY_MAP)

    weekday_avg = (
        df.groupby(["weekday", "weekday_name"], as_index=False)[target_col]
        .mean()
        .sort_values("weekday")
    )

    fig = px.bar(
        weekday_avg,
        x="weekday_name",
        y=target_col,
        title="Average Daily Sales by Weekday",
        labels={"weekday_name": "Weekday", target_col: "Average Unit Sales"},
    )
    return fig


def plot_monthly_seasonality(
    ts_daily: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
) -> go.Figure:
    """
    Plot average daily sales per calendar month.
    """
    df = _ensure_datetime(ts_daily[[date_col, target_col]], date_col)
    df["month"] = df[date_col].dt.month
    df["month_name"] = df["month"].map(_MONTH_LABELS)

    month_avg = (
        df.groupby(["month", "month_name"], as_index=False)[target_col]
        .mean()
        .sort_values("month")
    )

    fig = px.bar(
        month_avg,
        x="month_name",
        y=target_col,
        title="Average Daily Sales by Month",
        labels={"month_name": "Month", target_col: "Average Unit Sales"},
    )
    return fig


def plot_month_weekday_heatmap(
    ts_daily: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
) -> go.Figure:
    """
    Heatmap of average sales by month and weekday.
    """
    df = _ensure_datetime(ts_daily[[date_col, target_col]], date_col)
    df["weekday"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["weekday_name"] = df["weekday"].map(_WEEKDAY_MAP)
    df["month_name"] = df["month"].map(_MONTH_LABELS)

    mw_avg = (
        df.groupby(
            ["month", "month_name", "weekday", "weekday_name"],
            as_index=False,
        )[target_col]
        .mean()
    )

    fig = px.density_heatmap(
        mw_avg,
        x="weekday_name",
        y="month_name",
        z=target_col,
        title="Average Sales by Month and Weekday",
        labels={
            "weekday_name": "Weekday",
            "month_name": "Month",
            target_col: "Average Sales",
        },
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------

def plot_stl_components(
    ts_daily: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "unit_sales",
    period: int = 7,
) -> Dict[str, go.Figure]:
    """
    Decompose the daily series with STL and return component plots.

    Parameters
    ----------
    ts_daily:
        Daily aggregated frame.
    date_col:
        Date column.
    target_col:
        Target column.
    period:
        Seasonal period in days. For weekly seasonality use 7.

    Returns
    -------
    dict
        Dictionary with keys:
        - "observed_trend"
        - "seasonal"
        - "residual"
    """
    df = _ensure_datetime(ts_daily[[date_col, target_col]], date_col)

    series = (
        df.set_index(date_col)[target_col]
        .asfreq("D")
        .fillna(0)
    )

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    decomp_df = pd.DataFrame(
        {
            "date": series.index,
            "observed": series.values,
            "trend": result.trend.values,
            "seasonal": result.seasonal.values,
            "resid": result.resid.values,
        }
    )

    # Observed vs trend
    fig_obs_trend = go.Figure()
    fig_obs_trend.add_trace(
        go.Scatter(
            x=decomp_df["date"],
            y=decomp_df["observed"],
            mode="lines",
            name="Observed",
            opacity=0.3,
        )
    )
    fig_obs_trend.add_trace(
        go.Scatter(
            x=decomp_df["date"],
            y=decomp_df["trend"],
            mode="lines",
            name="Trend",
        )
    )
    fig_obs_trend.update_layout(
        title="Observed vs Trend (STL)",
        xaxis_title="Date",
        yaxis_title="Unit Sales",
    )

    # Seasonal component
    fig_seasonal = px.line(
        decomp_df,
        x="date",
        y="seasonal",
        title="Seasonal Component (STL)",
        labels={"date": "Date", "seasonal": "Seasonal effect"},
    )

    # Residuals
    fig_resid = px.line(
        decomp_df,
        x="date",
        y="resid",
        title="Residual Component (STL)",
        labels={"date": "Date", "resid": "Residual"},
    )

    return {
        "observed_trend": fig_obs_trend,
        "seasonal": fig_seasonal,
        "residual": fig_resid,
    }


__all__: Tuple[str, ...] = (
    "build_daily_series",
    "plot_daily_sales",
    "plot_rolling_averages",
    "plot_weekday_seasonality",
    "plot_monthly_seasonality",
    "plot_month_weekday_heatmap",
    "plot_stl_components",
)