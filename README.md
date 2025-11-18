# 📦 Corporación Favorita Grocery Sales Forecasting
A modular, clean, and scalable forecasting pipeline with Streamlit, Plotly, and Parquet preprocessing.

---

# 📚 Table of Contents
1. [Welcome](#-welcome)
2. [Quickstart](#-quickstart)
3. [VS Code Interpreter](#-vs-code-interpreter)
4. [EDA Notebooks](#-eda-notebooks)
5. [Overview EDA](#-1-overview-eda)
6. [Deep Dive EDA](#-2-deep-dive-eda)
7. [Seasonality Analysis](#-3-seasonality-analysis)

---

# 👋 Welcome

This repository contains a fully structured data-science workflow for the **Kaggle Corporación Favorita Sales Forecasting** challenge.

Included:

- ⚡ Efficient CSV → Parquet preprocessing
- 📂 Modular data loading & YAML config-based filtering
- 📊 Two-level Exploratory Data Analysis (Overview + Deep Dive)
- 🎨 Reusable Plotly charts (also usable in a future Streamlit app)
- 📈 Dedicated notebook for Seasonality & STL decomposition
- 🖥️ Future-ready Streamlit forecasting dashboard

All components follow a clean, maintainable architecture designed for reusability.

---

# 🚀 Quickstart

### macOS (zsh / bash)
```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -e ".[dev]"
```

### Windows PowerShell
```bash
python -m venv .venv; `
. .\.venv\Scripts\Activate.ps1; `
python -m pip install --upgrade pip; `
pip install -e ".[dev]"
```

### Windows CMD
```bash
python -m venv .venv && .\.venv\Scriptsctivate && python -m pip install --upgrade pip && pip install -e ".[dev]"
```

---

# 🧠 VS Code Interpreter

If VS Code does not automatically select the correct Python interpreter:

```bash
cmd + shift + p
```

Then search for:

```text
Python: Select Interpreter
```

![Interpreter](img/select-interpreter.png)

---

# 📊 EDA Notebooks

To run the project with visualization dependencies (and let notebooks save images into `img/reports/...`):

```bash
pip install -e ".[viz,dev]"
```

This enables:

- Automatic saving of EDA plots
- Plotly-based visualizations
- A clean environment for notebooks

---

# 🟦 1. Overview EDA

## 🏪 Store & Item Landscape

| Store distribution | Top stores by average sales |
| ------------------ | --------------------------- |
| ![](img/reports/eda_overview/stores_distribution.png) | ![](img/reports/eda_overview/top_stores_avg_sales.png) |

| Top 30 items | Unit sales distribution |
| ------------ | ----------------------- |
| ![](img/reports/eda_overview/top_30_items.png) | ![](img/reports/eda_overview/unit_sales_distribution.png) |

---

## 📈 Sales Patterns & Seasonality

| Total sales over time | Average sales by day of week |
| --------------------- | ---------------------------- |
| ![](img/reports/eda_overview/total_sales_over_time.png) | ![](img/reports/eda_overview/avg_sales_by_dayofweek.png) |

| Promotions vs. sales |
| -------------------- |
| ![](img/reports/eda_overview/promo_vs_sales.png) |

---

# 🟩 2. Deep Dive EDA

## 🎁 Items

| Top 40 families by total sales |
| ------------------------------ |
| ![](img/reports/eda_deepdive/items/items_family_top40.png) |

---

## 💵 Oil Prices

| Oil price time series |
| ---------------------- |
| ![](img/reports/eda_deepdive/oil/oil_price_timeseries.png) |

---

## 🎉 Holidays

| Holidays by locale |
| ------------------ |
| ![](img/reports/eda_deepdive/holidays_events/holidays_by_locale.png) |

---

## 🏙️ Stores

| Stores per city |
| --------------- |
| ![](img/reports/eda_deepdive/stores/stores_per_city.png) |

---

## 🛒 Train Dataset (Sales Deep Dive)

| Daily total sales | Unit sales histogram (sample) |
| ----------------- | ----------------------------- |
| ![](img/reports/eda_deepdive/train/train_daily_total_sales.png) | ![](img/reports/eda_deepdive/train/train_unit_sales_hist_sample.png) |

| Top 30 items by number of rows | Top 30 stores by number of rows |
| ------------------------------ | -------------------------------- |
| ![](img/reports/eda_deepdive/train/train_top30_items_by_rows.png) | ![](img/reports/eda_deepdive/train/train_top30_stores_by_rows.png) |

---

## 💳 Transactions

| Daily transaction totals |
| ------------------------ |
| ![](img/reports/eda_deepdive/transactions/transactions_daily_total.png) |

---

# 🟧 3. Seasonality Analysis

Understanding weekly, monthly, and long-term patterns in the sales time series.

This analysis includes:

- Daily aggregated sales
- Rolling averages (7, 30, 90 days)
- Weekly seasonality
- Monthly seasonality
- Month × weekday interaction (heatmap)
- STL decomposition (trend, seasonal, residual)

All generated in the dedicated seasonality notebook.

---

## 📆 Daily Sales & Trends

| Rolling averages |
| ---------------- |
| ![](img/reports/seasonality/seasonality_rolling_averages.png) |

---

## 📅 Weekly & Monthly Patterns

| Weekday pattern | Monthly pattern |
| --------------- | --------------- |
| ![](img/reports/seasonality/seasonality_weekday_pattern.png) | ![](img/reports/seasonality/seasonality_monthly_pattern.png) |

---

## 🔥 Month × Weekday Heatmap

| Month × weekday heatmap |
| ------------------------ |
| ![](img/reports/seasonality/seasonality_month_weekday_heatmap.png) |

---

## 🔍 STL Decomposition

| Observed vs trend | Seasonal component | Residuals |
| ----------------- | ------------------ | --------- |
| ![](img/reports/seasonality/seasonality_stl_observed_trend.png) | ![](img/reports/seasonality/seasonality_stl_seasonal_component.png) | ![](img/reports/seasonality/seasonality_stl_residuals.png) |

---

All plots are exported into:

```text
img/reports/eda_overview/
img/reports/eda_deepdive/
img/reports/seasonality/
```

These assets support both this README and the future Streamlit dashboard.
