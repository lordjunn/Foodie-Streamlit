# Code Guide

This guide explains how the Streamlit app is organized and where to make changes.

## Entry Point

- app.py
  - Initializes Streamlit settings and theme.
  - Manages scrape controls and session state.
  - Builds tabs and delegates rendering to tab modules.

## Data Flow

1. Scrape controls call services/data_service.py to load or scrape data.
2. Data is normalized and cached into food.csv via db.py.
3. The Data Explorer applies filters through ui/filters.py.
4. Tabs render views from the filtered data.

## Core Modules

- services/data_service.py
  - CSV bootstrap and incremental scraping logic.
  - Data filtering and CSV download export helper.

- db.py
  - CSV I/O and normalization.
  - Incremental month logic (earliest and latest edges).

- scraper.py
  - HTML scraping and parsing.

- helpers.py
  - Parsing helpers and meal type normalization.

- forecasting.py
  - Forecasting models and preparation logic.

- plots.py
  - Plotly chart builders.

## UI Modules

- ui/filters.py
  - Filter widgets and date range filtering.

- ui/metrics.py
  - KPI row at the top of the Data Explorer.

## Tabs

Each tab is a small renderer that accepts the filtered DataFrame.

- tabs/dashboard.py
- tabs/data_stats.py
- tabs/visualizations.py
- tabs/forecasting_tab.py
- tabs/food_gallery.py
- tabs/compare.py
- tabs/about.py

## Where to Change Things

- Add a new tab: create a renderer in tabs/ and import it in app.py.
- Adjust filters: edit ui/filters.py.
- Change scraping or cache behavior: edit services/data_service.py or db.py.
- Update charts: edit plots.py or the relevant tab module.
