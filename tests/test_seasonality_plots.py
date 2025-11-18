import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.plots.seasonality import (
    build_daily_series,
    plot_daily_sales,
    plot_rolling_averages,
    plot_weekday_seasonality,
    plot_monthly_seasonality,
    plot_month_weekday_heatmap,
    plot_stl_components,
)


# ---------------------------------------------------------------------
# Helper: create a minimal synthetic dataset for all tests
# ---------------------------------------------------------------------

def make_mock_data():
    """Creates a small, clean time series from Jan 1–30."""
    dates = pd.date_range("2020-01-01", "2020-01-30")
    df = pd.DataFrame({
        "date": dates.repeat(3),           # simulate 3 stores
        "unit_sales": [10, 20, 30] * len(dates)
    })
    return df


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_build_daily_series():
    df = make_mock_data()
    ts = build_daily_series(df)

    assert isinstance(ts, pd.DataFrame)
    assert "date" in ts.columns
    assert "unit_sales" in ts.columns
    assert len(ts) == 30
    assert ts["unit_sales"].iloc[0] == 60  # 10+20+30


def test_plot_daily_sales():
    df = build_daily_series(make_mock_data())
    fig = plot_daily_sales(df)
    assert isinstance(fig, go.Figure)


def test_plot_rolling_averages():
    df = build_daily_series(make_mock_data())
    fig = plot_rolling_averages(df)
    assert isinstance(fig, go.Figure)


def test_plot_weekday_seasonality():
    df = build_daily_series(make_mock_data())
    fig = plot_weekday_seasonality(df)
    assert isinstance(fig, go.Figure)


def test_plot_monthly_seasonality():
    df = build_daily_series(make_mock_data())
    fig = plot_monthly_seasonality(df)
    assert isinstance(fig, go.Figure)


def test_plot_month_weekday_heatmap():
    df = build_daily_series(make_mock_data())
    fig = plot_month_weekday_heatmap(df)
    assert isinstance(fig, go.Figure)


def test_plot_stl_components():
    df = build_daily_series(make_mock_data())
    results = plot_stl_components(df)

    assert isinstance(results, dict)
    assert set(results.keys()) == {"observed_trend", "seasonal", "residual"}

    for key, fig in results.items():
        assert isinstance(fig, go.Figure)