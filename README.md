# 🍜 Junn Food Log Scraper & Data Science Explorer

A full-featured **Streamlit** web application that scrapes, cleans, explores, and forecasts personal food spending data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://junn-foodie-data-science.streamlit.app/)

---

## ✨ Features

### 🏠 Dashboard
- Hero KPI metrics — total meals, total spend, average price, unique restaurants, date range
- Monthly spending trend (bar + line overlay)
- Top 10 restaurants and dishes (horizontal bar charts)
- Day-of-week distribution and meal type pie chart
- Personal records — most expensive, cheapest, most visited, most ordered

### 📋 Data Explorer
- Interactive filters: restaurant, meal type, date range, dish name search
- Toggle to ignore zero / near-zero prices
- CSV export of filtered data
- Statistical summaries per meal category (mean, median, variance, std dev, etc.)
- Monthly KPI breakdown with colour-coded deltas (Δ Avg, % Change)
- LOWESS monthly summaries by meal type
- Monthly KPI breakdown by meal type

### 📈 Visualizations
- Box plot of price by meal category
- QQ plot for normality check
- Price distribution histogram with normal curve overlay
- Prices over time (scatter + LOWESS trendlines by meal type)
- Eating habits heatmap (meal frequency intensity)

### 🔮 Forecasting
- **Prophet** — trend-only, robust to missing data
- **Linear Regression** — simple time-index trend extrapolation
- **Exponential Smoothing** — Holt-Winters additive trend
- **ARIMA** — autoregressive integrated moving average
- Model comparison chart with confidence intervals
- Individual model drill-down with forecast tables

### 🔄 Month-to-Month Comparison
- Select any two months and compare side-by-side
- KPI deltas: avg price, total meals, total spend, unique restaurants
- Overlaid price distribution histograms
- Top restaurants comparison
- Meal type pie chart comparison

### 🍽️ Food Gallery
- Three view modes: All Dishes (card grid), By Dish (evolution), By Restaurant (list)
- Sorting: date, price, alphabetical
- Pagination (24 per page)
- Optional price trend overlay (top 15 most ordered)

### ℹ️ About & Data Dictionary
- Project description, tech stack, and methodology
- Forecasting model reference table
- Full data dictionary for all columns

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | [Streamlit](https://streamlit.io/) |
| Scraping | BeautifulSoup, Requests |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Seaborn |
| Forecasting | Prophet, ARIMA (statsmodels), Exponential Smoothing, Scikit-learn |
| Statistics | SciPy |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/lordjunn/Foodie-Streamlit.git
cd Foodie-Streamlit
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Foodie-Streamlit/
├── .streamlit/
│   └── config.toml        # Theme configuration (dark/light mode)
├── app.py                  # Main Streamlit application
├── scraper.py              # Web scraper (with and without progress bar)
├── helpers.py              # Utility functions (parsing, normalization)
├── forecasting.py          # Forecasting models (Prophet, LR, ETS, ARIMA)
├── plots.py                # Plotly chart builders
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📖 Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date the meal was logged |
| `dish_name` | string | Name of the dish ordered |
| `restaurant_name` | string | Restaurant / stall name (from `[brackets]`) |
| `price` | string | Raw price text as scraped |
| `numeric_price` | float | Parsed numeric price in RM (`NaN` if unparseable) |
| `meal_type` | string | Breakfast, Lunch, Dinner, or Other |
| `description` | string | Dish description (inline HTML preserved) |
| `image_url` | string | Absolute URL to food image (if available) |

---

## 🗺️ Roadmap

- [ ] Project restructure (separate `config.py`, `models/`, `services/`, `tests/`)
- [ ] Type hints throughout all modules
- [ ] Logging (replace silent `pass` / `continue` in scraper)
- [ ] Unit tests (pytest) for scraper, price parsing, forecasting
- [ ] CI/CD (GitHub Actions) — lint, test, deploy
- [ ] Automatic model selection with time-series cross-validation & leaderboard
- [ ] Anomaly detection (flag unusually expensive meals / price jumps)
- [ ] Pre-loaded CSV snapshot for instant demo

---

## 📄 License

This project is for personal / educational use.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the framework
- [Plotly](https://plotly.com/) for interactive charts
- [Prophet](https://facebook.github.io/prophet/) for time-series forecasting

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/lordjunn">lordjunn</a>
</p>
