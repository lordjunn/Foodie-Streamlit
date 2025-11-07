import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from dateutil import parser
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="🍜 Junn Food Log Scraper & Data Explorer", layout="wide")
sns.set_theme(style="whitegrid")

# ---------- UTILS ----------
from helpers import (
    normalize_meal_type
)

# ---------- SCRAPER ----------
from scraper import scrape_data
from forecasting import forecast_prices

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

# ---------- APP ----------
st.title("🍜 Junn Food Log Scraper & Data Science Explorer")

# --- Sidebar: Scraping Settings ---
st.sidebar.header("Scrape Settings")

# Define options
year_options = [22, 23, 24, 25]
month_options = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Initialize session state
if "selected_years" not in st.session_state:
    st.session_state.selected_years = year_options.copy()
if "selected_months" not in st.session_state:
    st.session_state.selected_months = month_options.copy()

# --- YEARS ---
st.sidebar.subheader("Years")

col1, col2, col3 = st.sidebar.columns([3, 1, 1])
with col1:
    years = st.multiselect(
        "Select Years",
        options=year_options,
        default=st.session_state.selected_years,
        key="year_multiselect"
    )
with col2:
    if st.button("All", key="select_all_years"):
        st.session_state.selected_years = year_options.copy()
        st.session_state.year_multiselect = year_options.copy()
with col3:
    if st.button("Clear", key="clear_years"):
        st.session_state.selected_years = []
        st.session_state.year_multiselect = []

# --- MONTHS ---
st.sidebar.subheader("Months")

col4, col5, col6 = st.sidebar.columns([3, 1, 1])
with col4:
    months = st.multiselect(
        "Select Months",
        options=month_options,
        default=st.session_state.selected_months,
        key="month_multiselect"
    )
with col5:
    if st.button("All", key="select_all_months"):
        st.session_state.selected_months = month_options.copy()
        st.session_state.month_multiselect = month_options.copy()
with col6:
    if st.button("Clear", key="clear_months"):
        st.session_state.selected_months = []
        st.session_state.month_multiselect = []

# --- SCRAPE BUTTON ---
scrape_button = st.sidebar.button("🔍 Start Scraping")


if scrape_button:
    st.subheader("Scraping in Progress...")
    df = scrape_data(years, months)
    if not df.empty:
        st.success(f"✅ Scraped {len(df)} items!")
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download CSV", csv, "menu_items.csv", "text/csv")
        st.session_state['data'] = df
    else:
        st.warning("No data found.")

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

    # Date range filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)
        ]

    st.dataframe(filtered_df)
    
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
                fig_box = px.box(
                    filtered_df, 
                    x='meal_category', 
                    y='numeric_price',
                    color='meal_category', 
                    points="all",
                    hover_data=['dish_name', 'restaurant_name']
                )
                fig_box.update_layout(xaxis_title="Meal Category", yaxis_title="Price")
                st.plotly_chart(fig_box, width='stretch')

            # --- QQ Plot ---
            with colB:
                st.subheader("QQ Plot (Normality Check)")
                sorted_prices = np.sort(numeric_prices)
                theoretical_quantiles = np.sort(np.random.normal(numeric_prices.mean(), numeric_prices.std(), len(sorted_prices)))
                qq_df = pd.DataFrame({
                    "Theoretical Quantiles": theoretical_quantiles,
                    "Sample Quantiles": sorted_prices
                })
                fig_qq = px.scatter(
                    qq_df, 
                    x="Theoretical Quantiles", 
                    y="Sample Quantiles",
                    hover_data=[qq_df.index]
                )
                fig_qq.add_shape(
                    type="line",
                    x0=qq_df["Theoretical Quantiles"].min(),
                    y0=qq_df["Theoretical Quantiles"].min(),
                    x1=qq_df["Theoretical Quantiles"].max(),
                    y1=qq_df["Theoretical Quantiles"].max(),
                    line=dict(color="red", dash="dash")
                )
                fig_qq.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
                st.plotly_chart(fig_qq, width='stretch')

            # --- Histogram with Normal Overlay ---
            st.subheader("Price Distribution Histogram")
            numeric_prices = filtered_df['numeric_price'].dropna()
            fig_hist = px.histogram(numeric_prices, nbins=20, opacity=0.7, marginal=None)
            fig_hist.update_traces(name='Prices', marker_color='blue')

            # Normal curve overlay
            mean, std = numeric_prices.mean(), numeric_prices.std()
            x_vals = np.linspace(numeric_prices.min(), numeric_prices.max(), 200)
            bin_width = (numeric_prices.max() - numeric_prices.min()) / 20
            y_vals = norm.pdf(x_vals, mean, std) * len(numeric_prices) * bin_width

            fig_hist.add_scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='red'), name='Normal Curve')
            st.plotly_chart(fig_hist, width='stretch')

            # --- Prices Over Time ---
            st.subheader("📅 Prices Over Time")
            filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
            time_df = filtered_df.dropna(subset=['numeric_price', 'date']).copy()

            if time_df.empty:
                st.warning("No valid data for 'Prices Over Time' plot.")
            else:
                # Ensure numeric_price is float
                time_df['numeric_price'] = pd.to_numeric(time_df['numeric_price'], errors='coerce')
                time_df = time_df.dropna(subset=['numeric_price'])
                
                if len(time_df) < 2:
                    st.warning("Not enough data points to plot 'Prices Over Time'.")
                else:
                    fig_time = px.scatter(
                        time_df,
                        x='date',
                        y='numeric_price',
                        color='meal_category',
                        hover_data=['dish_name', 'restaurant_name'],
                        trendline='lowess',
                        title="Food Prices Over Time"
                    )
                    fig_time.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend_title="Meal Type"
                    )
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

            st.subheader("🔮 Forecast Next Month Prices")

            if not filtered_df.empty:
                periods = st.slider("Months to Forecast", 1, 12, 3)
                forecast_df, model = forecast_prices(filtered_df, periods=periods)

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines+markers', name='Actual'))
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.1)',
                    line=dict(width=0),
                    name='Confidence Interval'
                ))

                fig.update_layout(title='Predicted Monthly Average Price', xaxis_title='Date', yaxis_title='Price (RM)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to forecast.")


else:
    st.info("👈 Start by scraping some data first!")
