import streamlit as st


def render_about():
    st.header("ℹ️ About This Project")

    st.markdown(
        """
### 🍜 Junn Food Log Scraper & Data Science Explorer

A **Streamlit-powered** web application that scrapes, cleans, analyzes, and forecasts
personal food spending data from a custom food logging website.

**Built by:** [lordjunn](https://github.com/lordjunn)

---

### 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Data Scraping | BeautifulSoup, Requests |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Seaborn |
| Forecasting | Prophet, ARIMA, Exponential Smoothing, Linear Regression |
| Statistics | SciPy, Statsmodels, Scikit-learn |

---

### 📐 Methodology

1. **Data Collection** — Web scraping from custom HTML food logs hosted on GitHub Pages
2. **Data Cleaning** — Price parsing, meal type normalization, date standardization
3. **Exploratory Analysis** — Statistical summaries, LOWESS trendlines, distribution analysis (QQ plots, histograms)
4. **Forecasting** — Multiple time-series models trained on monthly aggregated price data with confidence intervals
5. **Visualization** — Interactive Plotly charts with filtering, sorting, and drill-down capabilities

---

### 🔮 Forecasting Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Prophet** | Facebook's time-series model with trend decomposition | Robust to missing data, handles holidays |
| **Linear Regression** | Simple trend extrapolation on time index | Baseline comparison, linear trends |
| **Exponential Smoothing** | Holt-Winters with additive trend | Short-term forecasts, smooth trends |
| **ARIMA** | Autoregressive Integrated Moving Average | Stationary series with autocorrelation |

---

### 📊 Key Features

- **Web Scraper** with progress UI and error handling for multi-month/year ingestion
- **Interactive Data Explorer** with filters by restaurant, meal type, date range, and search
- **Statistical Summaries** with per-category breakdowns and monthly KPIs with colour-coded deltas
- **LOWESS Trendlines** for smoothed price analysis by meal type
- **Multi-Model Forecasting** with comparison charts and confidence intervals
- **Food Gallery** with card / list / dish evolution views, sorting and pagination
- **Dashboard Overview** with hero metrics, top restaurants/dishes, and spending patterns
- **Month-to-Month Comparison** for side-by-side analysis of spending periods
        """
    )

    st.markdown("---")

    st.subheader("📖 Data Dictionary")
    st.markdown(
        """
| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date the meal was logged |
| `dish_name` | string | Name of the dish ordered |
| `restaurant_name` | string | Restaurant or food stall name (extracted from `[brackets]`) |
| `price` | string | Raw price text as scraped (may include "Free", strikethrough, etc.) |
| `numeric_price` | float | Parsed numeric price in RM; `NaN` if unparseable |
| `meal_type` | string | Meal category: Breakfast, Lunch, Dinner, or Other |
| `description` | string | Dish description with preserved inline HTML formatting |
| `image_url` | string | URL to the food image (if available) |

**Derived columns (computed in-app):**

| Column | Type | Description |
|--------|------|-------------|
| `meal_category` | string | Normalized meal type (Breakfast / Lunch / Dinner / Other) |
| `year_month` | string | Year-month period string (e.g., "2025-07") |
        """
    )

    st.markdown("---")
    st.caption(
        "Built with ❤️ using Streamlit · "
        "[Source Code](https://github.com/lordjunn/Foodie-Streamlit) · "
        "[Live Demo](https://junn-foodie-data-science.streamlit.app/)"
    )
