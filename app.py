import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import seaborn as sns
from datetime import datetime
from dateutil import parser

# ---------- CONFIG ----------
st.set_page_config(page_title="🍜 Junn Food Log Scraper & Data Explorer", layout="wide")
sns.set_theme(style="whitegrid")


TAHUN = [22, 23, 24, 25]
BULAN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# ---------- UTILS ----------
from helpers import (
    normalize_meal_type
)

# ---------- SCRAPER ----------
from scraper import scrape_data
from forecasting import forecast_prices
import plots

# [IMPROVEMENT] Caching
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------- FILTER & EXPLORE ----------
def filter_data(df, restaurants, meal_types, search):
    filtered_df = df.copy()
    if restaurants:
        filtered_df = filtered_df[filtered_df['restaurant_name'].isin(restaurants)]
    if meal_types:
        filtered_df = filtered_df[filtered_df['meal_type'].isin(meal_types)]
    if search:
        filtered_df = filtered_df[filtered_df['dish_name'].str.contains(search, case=False, na=False)]
    return filtered_df

def handle_scrape(df):
    if not df.empty:
        st.success(f"✅ Scraped {len(df)} items!")
        st.session_state['data'] = df
    else:
        st.warning("No data found.")

# ---------- APP ----------
st.title("🍜 Junn Food Log Scraper & Data Science Explorer")

# --- Sidebar: Scraping Settings ---
st.sidebar.header("Scrape Settings")
years = st.sidebar.multiselect("Years", TAHUN, default=[25])
months = st.sidebar.multiselect(
    "Months",
    BULAN,
    default=['Jul','Aug','Sep','Oct','Nov','Dec']
    #['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    #['Jan','Feb','Mar','Apr','May','Jun']
    #['Jul','Aug','Sep','Oct','Nov','Dec']
)
scrape_button = st.sidebar.button("🔍 Start Scraping")
st.sidebar.markdown("---")
scrape_all_button = st.sidebar.button("🌎 Scrape All (Ignore Filters)")

if scrape_button:
    st.subheader("Selective scraping in Progress...")
    handle_scrape(scrape_data(years, months)) 

if scrape_all_button:
    st.subheader("Super scraping in Progress...")
    handle_scrape(scrape_data(TAHUN, BULAN))  



# --- Data Explorer ---
st.header("📋 Data Explorer")
if 'data' in st.session_state:
    df = st.session_state['data'].copy()
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # --- Filters in 4 columns ---
    col1, col2, col3, col4 = st.columns([2,2,2,3])  # Adjust widths for better layout
    with col1:
        rest_filter = st.multiselect(
            "Filter by Restaurant", 
            options=sorted(df['restaurant_name'].dropna().unique().tolist())
        )
    with col2:
        meal_filter = st.multiselect(
            "Filter by Meal Type", 
            options=sorted(df['meal_type'].dropna().unique().tolist())
        )
    with col3:
        search = st.text_input("Search Dish Name")
    with col4:
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = st.date_input(
            "Filter by Date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # --- Apply filters ---
    filtered_df = filter_data(df, rest_filter, meal_filter, search)

    # NEW — Zero-value filter
    ignore_zero = st.checkbox("Ignore zero or near-zero prices (numeric_price < 0.01)", value=False)
    if ignore_zero:
        filtered_df = filtered_df[filtered_df['numeric_price'] >= 0.01]

    # Date range filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)
        ]

    # [IMPROVEMENT] KPI Metrics
    if not filtered_df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Meals", f"{len(filtered_df)}")
        if 'numeric_price' in filtered_df.columns and not filtered_df['numeric_price'].dropna().empty:
             m2.metric("Avg Price", f"RM {filtered_df['numeric_price'].mean():.2f}")
             m3.metric("Highest Price", f"RM {filtered_df['numeric_price'].max():.2f}")
        else:
             m2.metric("Avg Price", "-")
             m3.metric("Highest Price", "-")
        m4.metric("Most Visited", filtered_df['restaurant_name'].mode()[0] if not filtered_df.empty else "-")
        st.markdown("---")

    # [IMPROVEMENT] Create Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Data & Stats", "📈 Visualizations", "🔮 Forecasting"])

    with tab1:
        st.dataframe(filtered_df)
        csv = convert_df(filtered_df)
        st.download_button("💾 Download Filtered CSV", csv, "filtered_menu_items.csv", "text/csv")
    
    # --- Quantitative Summary ---
        st.subheader("📊 Quantitative Summary")
        df_stats = filtered_df.dropna(subset=['numeric_price']).copy()
        if not df_stats.empty:
            df_stats['meal_category'] = df_stats['meal_type'].apply(normalize_meal_type)
            grouped_stats = df_stats.groupby('meal_category')['numeric_price'].agg(
                ['mean', 'median', 'min', 'max', 'var', 'std', 'count']
            ).rename(columns={
                'mean':'Mean','median':'Median','min':'Min','max':'Max',
                'var':'Variance','std':'Std Dev','count':'Count'
            })

            total_stats = pd.DataFrame({
                'Mean':[df_stats['numeric_price'].mean()],
                'Median':[df_stats['numeric_price'].median()],
                'Min':[df_stats['numeric_price'].min()],
                'Max':[df_stats['numeric_price'].max()],
                'Variance':[df_stats['numeric_price'].var()],
                'Std Dev':[df_stats['numeric_price'].std()],
                'Count':[df_stats['numeric_price'].count()]
            }, index=['🍽️ Grand Total'])

            full_stats = pd.concat([grouped_stats, total_stats])
            st.dataframe(full_stats.style.format({
                'Mean': '{:.2f}', 'Median': '{:.2f}', 'Min': '{:.2f}', 'Max': '{:.2f}',
                'Variance': '{:.2f}', 'Std Dev': '{:.2f}', 'Count': '{:,.0f}'
            }))

    # --- Visualizations ---
    with tab2:
        st.header("📈 Interactive Visualizations")

        if not filtered_df.empty and 'numeric_price' in filtered_df.columns:
            filtered_df['meal_category'] = filtered_df['meal_type'].apply(normalize_meal_type)
            numeric_prices = filtered_df['numeric_price'].dropna()

            if not numeric_prices.empty:
                # --- Box Plot & QQ Plot side by side ---
                colA, colB = st.columns(2)

                # --- Box Plot ---
                with colA:
                    st.subheader("Box Plot: Price by Meal Category")
                    fig_box = plots.plot_box_price_by_category(filtered_df)
                    st.plotly_chart(fig_box, width='stretch')

                # --- QQ Plot ---
                with colB:
                    st.subheader("QQ Plot (Normality Check)")
                    fig_qq = plots.plot_qq_normality_check(numeric_prices)
                    st.plotly_chart(fig_qq, width='stretch')

                # --- Histogram with Normal Overlay ---
                st.subheader("Price Distribution Histogram")
                fig_hist = plots.plot_price_distribution(numeric_prices)
                st.plotly_chart(fig_hist, width='stretch')

                # --- Prices Over Time ---
                st.subheader("📅 Prices Over Time")
                # Ensure date is datetime for plotting
                filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
                fig_time = plots.plot_prices_over_time(filtered_df)
                
                if fig_time:
                    st.plotly_chart(fig_time, width='stretch')
                    
                    # --- LOWESS Monthly Summary per Meal Type (clean layout) ---
                    st.subheader("📊 Monthly LOWESS Summary (by Meal Type)")

                    lowess_traces = [t for t in fig_time.data if getattr(t, "mode", None) == "lines"]

                    if len(lowess_traces) == 0:
                        st.info("No LOWESS trendlines found — try enabling trendline='lowess' in the plot.")
                    else:
                        # Make pairs of columns (2x2 layout)
                        cols = st.columns(2)
                        col_index = 0

                        for trace in lowess_traces:
                            # Extract meal type
                            meal_type = trace.name.replace("(lowess)", "").strip()
                            lowess_x = pd.to_datetime(trace.x)
                            lowess_y = trace.y

                            lowess_df = pd.DataFrame({
                                'date': lowess_x,
                                'lowess_price': lowess_y
                            })
                            lowess_df['year_month'] = lowess_df['date'].dt.to_period('M').astype(str)

                            monthly_summary = lowess_df.groupby('year_month')['lowess_price'].agg(
                                ['min', 'max', 'mean']
                            ).rename(columns={
                                'min': 'LOWESS Min',
                                'max': 'LOWESS Max',
                                'mean': 'LOWESS Avg'
                            }).round(2)

                            monthly_summary['Δ Avg'] = monthly_summary['LOWESS Avg'].diff().round(2)
                            monthly_summary['% Change Avg'] = (
                                monthly_summary['LOWESS Avg'].pct_change() * 100
                            ).round(2)

                            def highlight_change(val):
                                if pd.isna(val):
                                    return ''
                                color = 'green' if val > 0 else ('red' if val < 0 else 'gray')
                                return f'color: {color}; font-weight: bold;'

                            # Use the current column
                            with cols[col_index]:
                                st.markdown(f"### 🍽️ {meal_type}")
                                st.dataframe(
                                    monthly_summary.style.format({
                                        'LOWESS Min': '{:.2f}',
                                        'LOWESS Max': '{:.2f}',
                                        'LOWESS Avg': '{:.2f}',
                                        'Δ Avg': '{:+.2f}',
                                        '% Change Avg': '{:+.2f}%'
                                    }).applymap(highlight_change, subset=['Δ Avg', '% Change Avg'])
                                )

                            # Alternate between left (0) and right (1) column
                            col_index = (col_index + 1) % 2
                            if col_index == 0:
                                cols = st.columns(2)  # Start a new row of 2 columns
                else:
                    st.warning("Not enough data points to plot 'Prices Over Time'.")

                # [IMPROVEMENT] Activity Heatmap
                st.subheader("📅 Eating Habits Heatmap")
                fig_cal = plots.plot_calendar_heatmap(filtered_df)
                st.plotly_chart(fig_cal, use_container_width=True)

    with tab3:
        st.subheader("🔮 Forecast Next Month Prices")

        # Assume 'filtered_df' exists from your previous filtering
        if not filtered_df.empty:
            periods = st.slider("Months to Forecast", 1, 24, 3)
            forecast_df, model = forecast_prices(filtered_df, periods=periods, smooth=True)

            # Plot
            fig = plots.plot_forecast(forecast_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to forecast.")


else:
    st.info("👈 Start by scraping some data first!")
