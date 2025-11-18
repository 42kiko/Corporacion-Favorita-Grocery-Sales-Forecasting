import pytest
import pandas as pd
from plotly.graph_objects import Figure

from src.plots.overview import (
    plot_store_distribution,
    plot_top_stores_avg_sales,
    plot_top_items,
    plot_unit_sales_distribution,
    plot_total_sales_over_time,
    plot_avg_sales_by_weekday,
    plot_promo_vs_sales,
)


# ---------------------------------------------------------
# Fixtures: Small synthetic datasets
# ---------------------------------------------------------
@pytest.fixture
def train_df():
    """Synthetic train data with minimal valid structure."""
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=5),
        "store_nbr": [1, 1, 2, 2, 3],
        "item_nbr": [101, 102, 101, 103, 101],
        "unit_sales": [5, 3, 2, 7, 1],
        "onpromotion": [0, 1, 0, 0, 1],
    })


@pytest.fixture
def items_meta_df():
    """Synthetic metadata for items."""
    return pd.DataFrame({
        "item_nbr": [101, 102, 103],
        "family": ["Beverages", "Snacks", "Produce"],
        "class": [10, 20, 30],
        "perishable": [0, 0, 1],
    })


# ---------------------------------------------------------
# 1. Test: Store distribution
# ---------------------------------------------------------
def test_plot_store_distribution(train_df):
    fig = plot_store_distribution(train_df)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------
# 2. Test: Top stores by avg sales
# ---------------------------------------------------------
def test_plot_top_stores_avg_sales(train_df):
    fig = plot_top_stores_avg_sales(train_df, top_n=2)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------
# 3. Test: Top items with metadata
# ---------------------------------------------------------
def test_plot_top_items(train_df, items_meta_df):
    fig = plot_top_items(train_df, items_meta_df, top_n=3)
    assert isinstance(fig, Figure)


# Edge-case: wrong metadata should raise error
def test_plot_top_items_invalid_meta(train_df):
    bad_meta = pd.DataFrame({"item_nbr": [101]})  # missing required columns
    with pytest.raises(ValueError):
        plot_top_items(train_df, bad_meta)


# ---------------------------------------------------------
# 4. Test: Item sales distribution histogram
# ---------------------------------------------------------
def test_plot_unit_sales_distribution(train_df):
    fig = plot_unit_sales_distribution(train_df)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------
# 5. Test: Total sales over time
# ---------------------------------------------------------
def test_plot_total_sales_over_time(train_df):
    fig = plot_total_sales_over_time(train_df)
    assert isinstance(fig, Figure)


def test_plot_total_sales_over_time_filtered(train_df):
    fig = plot_total_sales_over_time(train_df, date_range=("2021-01-01", "2021-01-03"))
    assert isinstance(fig, Figure)


# ---------------------------------------------------------
# 6. Test: Sales by weekday
# ---------------------------------------------------------
def test_plot_avg_sales_by_weekday(train_df):
    fig = plot_avg_sales_by_weekday(train_df)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------
# 7. Test: Promotion vs sales
# ---------------------------------------------------------
def test_plot_promo_vs_sales(train_df):
    fig = plot_promo_vs_sales(train_df)
    assert isinstance(fig, Figure)