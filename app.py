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


TAHUN = [22, 23, 24, 25, 26]
BULAN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# ---------- UTILS ----------
from helpers import (
    normalize_meal_type
)

# ---------- SCRAPER ----------
from scraper import scrape_data, scrape_data_raw
import plotly.express as px
import plotly.graph_objects as go
from forecasting import (
    forecast_prices,
    forecast_linear_regression,
    forecast_exponential_smoothing,
    forecast_arima
)
import plots

# [IMPROVEMENT] Caching
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# [IMPROVEMENT] 12-hour cache for scraped data
@st.cache_data(ttl=43200, show_spinner="Loading data (first run may take a minute)…")
def load_cached_data(years_tuple, months_tuple):
    """Cache-safe scraping — no Streamlit UI widgets."""
    return scrape_data_raw(list(years_tuple), list(months_tuple))

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
        st.success(f"✅ Loaded {len(df)} items!")
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
clear_cache = st.sidebar.button("🗑️ Clear Cache")

if clear_cache:
    load_cached_data.clear()
    st.toast("Cache cleared!")

if scrape_button:
    handle_scrape(load_cached_data(tuple(years), tuple(months)))

if scrape_all_button:
    handle_scrape(load_cached_data(tuple(TAHUN), tuple(BULAN)))



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
    tab_dash, tab1, tab2, tab3, tab4, tab_compare, tab_about = st.tabs([
        "🏠 Dashboard", "📋 Data & Stats", "📈 Visualizations",
        "🔮 Forecasting", "🍽️ Food Images", "🔄 Compare", "ℹ️ About"
    ])

    # ──────────────────────────────────────────────────────
    # TAB — Dashboard Overview
    # ──────────────────────────────────────────────────────
    with tab_dash:
        st.header("🏠 Dashboard Overview")

        if filtered_df.empty:
            st.info("No data to display. Adjust your filters or scrape more data.")
        else:
            # --- Hero Metrics ---
            prices_valid = filtered_df['numeric_price'].dropna()
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Total Meals", f"{len(filtered_df):,}")
            d2.metric("Total Spend", f"RM {prices_valid.sum():,.2f}" if not prices_valid.empty else "—")
            d3.metric("Avg per Meal", f"RM {prices_valid.mean():.2f}" if not prices_valid.empty else "—")
            d4.metric("Unique Restaurants", filtered_df['restaurant_name'].nunique())
            date_min = filtered_df['date'].min()
            date_max = filtered_df['date'].max()
            if pd.notna(date_min) and pd.notna(date_max):
                d5.metric("Date Range", f"{date_min.strftime('%b %Y')} – {date_max.strftime('%b %Y')}")
            else:
                d5.metric("Date Range", "—")

            st.markdown("---")

            # --- Monthly Spending Trend (bar + line) ---
            st.subheader("📈 Monthly Spending Trend")
            trend_df = filtered_df.dropna(subset=['numeric_price', 'date']).copy()
            if not trend_df.empty:
                trend_df['year_month'] = trend_df['date'].dt.to_period('M').astype(str)
                monthly_agg = trend_df.groupby('year_month').agg(
                    total=('numeric_price', 'sum'),
                    avg=('numeric_price', 'mean'),
                    count=('numeric_price', 'count')
                ).reset_index()

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=monthly_agg['year_month'], y=monthly_agg['total'],
                    name='Total Spend (RM)', marker_color='#636EFA'
                ))
                fig_trend.add_trace(go.Scatter(
                    x=monthly_agg['year_month'], y=monthly_agg['avg'],
                    name='Avg Price (RM)', yaxis='y2',
                    line=dict(color='#EF553B', width=3), mode='lines+markers'
                ))
                fig_trend.update_layout(
                    yaxis=dict(title='Total Spend (RM)'),
                    yaxis2=dict(title='Avg Price (RM)', overlaying='y', side='right'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            # --- Top Restaurants + Top Dishes ---
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("🏆 Top 10 Restaurants")
                top_rest = filtered_df['restaurant_name'].value_counts().head(10).reset_index()
                top_rest.columns = ['Restaurant', 'Visits']
                fig_rest = px.bar(top_rest, x='Visits', y='Restaurant', orientation='h',
                                  color='Visits', color_continuous_scale='Blues')
                fig_rest.update_layout(yaxis=dict(autorange='reversed'), height=400, showlegend=False)
                st.plotly_chart(fig_rest, use_container_width=True)

            with col_r:
                st.subheader("🍜 Top 10 Most Ordered Dishes")
                top_dish = filtered_df['dish_name'].value_counts().head(10).reset_index()
                top_dish.columns = ['Dish', 'Orders']
                fig_dish = px.bar(top_dish, x='Orders', y='Dish', orientation='h',
                                  color='Orders', color_continuous_scale='Oranges')
                fig_dish.update_layout(yaxis=dict(autorange='reversed'), height=400, showlegend=False)
                st.plotly_chart(fig_dish, use_container_width=True)

            # --- Day of Week + Meal Type ---
            col_dow, col_mt = st.columns(2)
            with col_dow:
                st.subheader("📅 Meals by Day of Week")
                dow_df = filtered_df.dropna(subset=['date']).copy()
                if not dow_df.empty:
                    dow_df['day_of_week'] = dow_df['date'].dt.day_name()
                    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    dow_counts = dow_df['day_of_week'].value_counts().reindex(dow_order).fillna(0)
                    fig_dow = px.bar(x=dow_counts.index, y=dow_counts.values,
                                     labels={'x': 'Day', 'y': 'Meals'},
                                     color=dow_counts.values, color_continuous_scale='Viridis')
                    fig_dow.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_dow, use_container_width=True)

            with col_mt:
                st.subheader("🍽️ Meal Type Distribution")
                mt_counts = filtered_df['meal_type'].value_counts()
                if not mt_counts.empty:
                    fig_mt = px.pie(values=mt_counts.values, names=mt_counts.index, hole=0.4)
                    fig_mt.update_layout(height=350)
                    st.plotly_chart(fig_mt, use_container_width=True)

            # --- Personal Records ---
            st.markdown("---")
            st.subheader("🏅 Personal Records")
            if not prices_valid.empty:
                rec1, rec2, rec3, rec4 = st.columns(4)
                most_exp = filtered_df.loc[filtered_df['numeric_price'].idxmax()]
                cheapest_mask = filtered_df['numeric_price'] > 0
                if cheapest_mask.any():
                    cheapest = filtered_df.loc[filtered_df.loc[cheapest_mask, 'numeric_price'].idxmin()]
                else:
                    cheapest = filtered_df.loc[filtered_df['numeric_price'].idxmin()]

                rec1.metric("💰 Most Expensive", f"RM {most_exp['numeric_price']:.2f}",
                            most_exp['dish_name'][:30])
                rec2.metric("🪙 Cheapest", f"RM {cheapest['numeric_price']:.2f}",
                            cheapest['dish_name'][:30])

                most_visited = filtered_df['restaurant_name'].value_counts()
                if not most_visited.empty:
                    rec3.metric("🏠 Most Visited", most_visited.index[0][:20],
                                f"{most_visited.iloc[0]} visits")

                most_ordered = filtered_df['dish_name'].value_counts()
                if not most_ordered.empty:
                    rec4.metric("🔁 Most Ordered", most_ordered.index[0][:20],
                                f"{most_ordered.iloc[0]} times")

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

        # --- Monthly KPI Breakdown Table ---
        st.subheader("📅 Monthly KPI Breakdown")
        if not filtered_df.empty and 'numeric_price' in filtered_df.columns:
            monthly_df = filtered_df.dropna(subset=['numeric_price', 'date']).copy()
            monthly_df['year_month'] = monthly_df['date'].dt.to_period('M').astype(str)
            
            # Aggregate per month
            monthly_kpi = monthly_df.groupby('year_month').agg(
                Total_Meals=('numeric_price', 'count'),
                Avg_Price=('numeric_price', 'mean'),
                Min_Price=('numeric_price', 'min'),
                Max_Price=('numeric_price', 'max'),
                Most_Visited=('restaurant_name', lambda x: x.mode().iloc[0] if not x.mode().empty else '-')
            ).reset_index()
            
            monthly_kpi['Δ Avg'] = monthly_kpi['Avg_Price'].diff().round(2)
            monthly_kpi['% Change Avg'] = (monthly_kpi['Avg_Price'].pct_change() * 100).round(2)
            monthly_kpi = monthly_kpi.set_index('year_month')
            
            def highlight_change(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val > 0 else ('red' if val < 0 else 'gray')
                return f'color: {color}; font-weight: bold;'
            
            st.dataframe(
                monthly_kpi.style.format({
                    'Total_Meals': '{:,.0f}',
                    'Avg_Price': 'RM {:.2f}',
                    'Min_Price': 'RM {:.2f}',
                    'Max_Price': 'RM {:.2f}',
                    'Δ Avg': '{:+.2f}',
                    '% Change Avg': '{:+.2f}%'
                }).applymap(highlight_change, subset=['Δ Avg', '% Change Avg'])
            )

        # --- LOWESS Monthly Summary per Meal Type ---
        st.subheader("📊 Monthly LOWESS Summary (by Meal Type)")
        if not filtered_df.empty and 'numeric_price' in filtered_df.columns:
            filtered_df['meal_category'] = filtered_df['meal_type'].apply(normalize_meal_type)
            filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
            fig_time = plots.plot_prices_over_time(filtered_df)
            
            if fig_time:
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
                st.info("Not enough data points to generate LOWESS summary.")

        # --- Monthly KPI Breakdown by Meal Type ---
        st.subheader("📈 Monthly KPI by Meal Type")
        if not filtered_df.empty and 'numeric_price' in filtered_df.columns:
            mt_df = filtered_df.dropna(subset=['numeric_price', 'date']).copy()
            mt_df['meal_category'] = mt_df['meal_type'].apply(normalize_meal_type)
            mt_df['year_month'] = mt_df['date'].dt.to_period('M').astype(str)

            grouped_mt = mt_df.groupby(['meal_category', 'year_month']).agg(
                Total_Meals=('numeric_price', 'count'),
                Avg_Price=('numeric_price', 'mean'),
                Min_Price=('numeric_price', 'min'),
                Max_Price=('numeric_price', 'max'),
                Most_Visited=('restaurant_name', lambda x: x.mode().iloc[0] if not x.mode().empty else '-')
            ).reset_index()

            meal_types = grouped_mt['meal_category'].unique()
            cols = st.columns(2)
            col_index = 0

            def highlight_change(val):
                if pd.isna(val):
                    return ''
                color = 'green' if val > 0 else ('red' if val < 0 else 'gray')
                return f'color: {color}; font-weight: bold;'

            for meal in meal_types:
                with cols[col_index]:
                    st.markdown(f"### 🍽️ {meal}")
                    mdf = grouped_mt[grouped_mt['meal_category'] == meal].set_index('year_month').drop(columns=['meal_category'])
                    mdf['Δ Avg'] = mdf['Avg_Price'].diff().round(2)
                    mdf['% Change Avg'] = (mdf['Avg_Price'].pct_change() * 100).round(2)

                    st.dataframe(
                        mdf.style.format({
                            'Total_Meals': '{:,.0f}',
                            'Avg_Price': 'RM {:.2f}',
                            'Min_Price': 'RM {:.2f}',
                            'Max_Price': 'RM {:.2f}',
                            'Δ Avg': '{:+.2f}',
                            '% Change Avg': '{:+.2f}%'
                        }).applymap(highlight_change, subset=['Δ Avg', '% Change Avg'])
                    )

                col_index = (col_index + 1) % 2
                if col_index == 0:
                    cols = st.columns(2)

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
                else:
                    st.warning("Not enough data points to plot 'Prices Over Time'.")

                # [IMPROVEMENT] Activity Heatmap
                st.subheader("📅 Eating Habits Heatmap")
                fig_cal = plots.plot_calendar_heatmap(filtered_df)
                st.plotly_chart(fig_cal, use_container_width=True)

    with tab3:
        st.subheader("🔮 Forecast Next Month Prices")

        if not filtered_df.empty:
            periods = st.slider("Months to Forecast", 1, 24, 3)
            
            # Model selection
            st.markdown("### Select Models to Compare")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                use_prophet = st.checkbox("Prophet", value=True)
            with col_m2:
                use_linear = st.checkbox("Linear Regression", value=True)
            with col_m3:
                use_exp = st.checkbox("Exponential Smoothing", value=True)
            with col_m4:
                use_arima = st.checkbox("ARIMA", value=True)
            
            forecasts = {}
            errors = []
            
            # Run selected models
            if use_prophet:
                try:
                    forecast_df, _ = forecast_prices(filtered_df, periods=periods, smooth=True)
                    forecasts['Prophet'] = forecast_df
                except Exception as e:
                    errors.append(f"Prophet: {e}")
            
            if use_linear:
                try:
                    forecast_df, _ = forecast_linear_regression(filtered_df, periods=periods)
                    forecasts['Linear Regression'] = forecast_df
                except Exception as e:
                    errors.append(f"Linear Regression: {e}")
            
            if use_exp:
                try:
                    forecast_df, _ = forecast_exponential_smoothing(filtered_df, periods=periods)
                    forecasts['Exponential Smoothing'] = forecast_df
                except Exception as e:
                    errors.append(f"Exponential Smoothing: {e}")
            
            if use_arima:
                try:
                    forecast_df, _ = forecast_arima(filtered_df, periods=periods)
                    forecasts['ARIMA'] = forecast_df
                except Exception as e:
                    errors.append(f"ARIMA: {e}")
            
            # Show errors if any
            if errors:
                with st.expander("⚠️ Model Errors", expanded=False):
                    for err in errors:
                        st.warning(err)
            
            # Plot comparison if we have forecasts
            if forecasts:
                st.markdown("### 📊 Model Comparison")
                fig_compare = plots.plot_forecast_comparison(forecasts)
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Individual model plots
                st.markdown("### 📈 Individual Model Forecasts")
                for model_name, forecast_df in forecasts.items():
                    with st.expander(f"{model_name} Details", expanded=False):
                        fig = plots.plot_forecast(forecast_df)
                        fig.update_layout(title=f"{model_name} Forecast")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast table
                        st.dataframe(
                            forecast_df.style.format({
                                'y': '{:.2f}',
                                'yhat': '{:.2f}',
                                'yhat_lower': '{:.2f}',
                                'yhat_upper': '{:.2f}'
                            })
                        )
            else:
                st.warning("No models could generate forecasts. Try adjusting your data filters.")
        else:
            st.info("Not enough data to forecast.")

    # ──────────────────────────────────────────────────────
    # TAB 4 — Food Gallery / Food Images
    # ──────────────────────────────────────────────────────
    with tab4:
        st.header("🍽️ Food Gallery")

        if filtered_df.empty:
            st.info("No data to display. Adjust filters or scrape more data.")
        else:
            # Backward compat: ensure image_url column exists
            if 'image_url' not in filtered_df.columns:
                filtered_df['image_url'] = None

            # --- Controls ---
            gc1, gc2 = st.columns(2)
            with gc1:
                view_mode = st.radio(
                    "View Mode",
                    ["All Dishes", "By Dish", "By Restaurant"],
                    horizontal=True,
                    key="gallery_view",
                )
            with gc2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Date (Newest)", "Date (Oldest)",
                     "Price (High → Low)", "Price (Low → High)",
                     "A → Z", "Z → A"],
                    key="gallery_sort",
                )

            gallery_df = filtered_df.copy()

            # --- View-specific filters ---
            if view_mode == "By Dish":
                dish_opts = sorted(gallery_df['dish_name'].dropna().unique())
                if dish_opts:
                    sel_dish = st.selectbox("Select Dish", dish_opts, key="gal_dish")
                    gallery_df = gallery_df[gallery_df['dish_name'] == sel_dish]
                else:
                    st.warning("No dishes found.")
            elif view_mode == "By Restaurant":
                rest_opts = sorted(gallery_df['restaurant_name'].dropna().unique())
                if rest_opts:
                    sel_rest = st.selectbox("Select Restaurant", rest_opts, key="gal_rest")
                    gallery_df = gallery_df[gallery_df['restaurant_name'] == sel_rest]
                else:
                    st.warning("No restaurants found.")

            # --- Sorting ---
            _sort_cfg = {
                "Date (Newest)":      ("date", False),
                "Date (Oldest)":      ("date", True),
                "Price (High → Low)": ("numeric_price", False),
                "Price (Low → High)": ("numeric_price", True),
                "A → Z":              ("dish_name", True),
                "Z → A":              ("dish_name", False),
            }
            _scol, _sasc = _sort_cfg[sort_by]
            gallery_df = gallery_df.sort_values(
                _scol, ascending=_sasc, na_position="last"
            )

            # --- Optional Price Trends (max 15 lines) ---
            show_trend = st.checkbox("📈 Show Price Trends", key="gal_trend")
            if show_trend and not gallery_df.empty:
                _tdf = gallery_df.dropna(subset=["numeric_price", "date"]).copy()
                _tdf["label"] = (
                    _tdf["dish_name"] + " [" + _tdf["restaurant_name"] + "]"
                )
                _top = _tdf["label"].value_counts().head(15).index
                _tdf = _tdf[_tdf["label"].isin(_top)]
                if not _tdf.empty:
                    _fig = px.line(
                        _tdf.sort_values("date"),
                        x="date",
                        y="numeric_price",
                        color="label",
                        markers=True,
                        title=f"Price Trends (Top {len(_top)} most ordered)",
                    )
                    _fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price (RM)",
                        legend_title="Dish [Restaurant]",
                        height=500,
                    )
                    st.plotly_chart(_fig, use_container_width=True)

            # --- Gallery display ---
            st.caption(f"Showing {len(gallery_df)} entries")

            if gallery_df.empty:
                st.info("No items match your current selection.")
            else:
                # Pagination
                PER_PAGE = 24
                total_pages = max(1, -(-len(gallery_df) // PER_PAGE))
                page = st.number_input(
                    "Page", 1, total_pages, 1, key="gal_page"
                )
                start = (page - 1) * PER_PAGE
                page_df = gallery_df.iloc[start : start + PER_PAGE]

                # ---------- By Dish: evolution view ----------
                if view_mode == "By Dish":
                    for rest_name, grp in page_df.groupby(
                        "restaurant_name", sort=False
                    ):
                        st.subheader(f"📍 {rest_name}")
                        grp = grp.sort_values("date")
                        with st.container(height=420):
                            for _, r in grp.iterrows():
                                ci, ct = st.columns([1, 2])
                                with ci:
                                    img = r.get("image_url")
                                    if pd.notna(img) and img:
                                        st.image(img, use_container_width=True)
                                    else:
                                        st.markdown("🍽️ *No image*")
                                with ct:
                                    d = (
                                        r["date"].strftime("%d %b %Y")
                                        if pd.notna(r.get("date"))
                                        else "—"
                                    )
                                    p = (
                                        f"RM {r['numeric_price']:.2f}"
                                        if pd.notna(r.get("numeric_price"))
                                        else r.get("price", "—")
                                    )
                                    st.markdown(
                                        f"**{d}** · {p} · _{r.get('meal_type', '')}_"
                                    )
                                    desc = r.get("description", "")
                                    if pd.notna(desc) and desc not in (
                                        "No description",
                                        "",
                                    ):
                                        st.markdown(
                                            desc, unsafe_allow_html=True
                                        )
                                st.divider()

                # ---------- By Restaurant: list view ----------
                elif view_mode == "By Restaurant":
                    with st.container(height=600):
                        for _, r in page_df.iterrows():
                            ci, ct = st.columns([1, 2])
                            with ci:
                                img = r.get("image_url")
                                if pd.notna(img) and img:
                                    st.image(img, use_container_width=True)
                                else:
                                    st.markdown("🍽️ *No image*")
                            with ct:
                                d = (
                                    r["date"].strftime("%d %b %Y")
                                    if pd.notna(r.get("date"))
                                    else "—"
                                )
                                p = (
                                    f"RM {r['numeric_price']:.2f}"
                                    if pd.notna(r.get("numeric_price"))
                                    else r.get("price", "—")
                                )
                                st.markdown(f"**{r['dish_name']}** · {p}")
                                st.caption(
                                    f"📅 {d} · {r.get('meal_type', '')}"
                                )
                                desc = r.get("description", "")
                                if pd.notna(desc) and desc not in (
                                    "No description",
                                    "",
                                ):
                                    st.markdown(desc, unsafe_allow_html=True)
                            st.divider()

                # ---------- All Dishes: card grid ----------
                else:
                    COLS = 3
                    with st.container(height=600):
                        for i in range(0, len(page_df), COLS):
                            cols = st.columns(COLS)
                            for j, col in enumerate(cols):
                                if i + j >= len(page_df):
                                    break
                                row = page_df.iloc[i + j]
                                with col:
                                    img = row.get("image_url")
                                    if pd.notna(img) and img:
                                        st.image(
                                            img, use_container_width=True
                                        )
                                    d = (
                                        row["date"].strftime("%d %b %Y")
                                        if pd.notna(row.get("date"))
                                        else "—"
                                    )
                                    p = (
                                        f"RM {row['numeric_price']:.2f}"
                                        if pd.notna(row.get("numeric_price"))
                                        else row.get("price", "—")
                                    )
                                    st.markdown(f"**{row['dish_name']}**")
                                    st.caption(
                                        f"📍 {row['restaurant_name']} · {p} · {d}"
                                    )

    # ──────────────────────────────────────────────────────
    # TAB — Month-to-Month Comparison
    # ──────────────────────────────────────────────────────
    with tab_compare:
        st.header("🔄 Month-to-Month Comparison")

        comp_df = filtered_df.dropna(subset=['numeric_price', 'date']).copy()
        if not comp_df.empty:
            comp_df['year_month'] = comp_df['date'].dt.to_period('M').astype(str)
            available_months = sorted(comp_df['year_month'].unique())

            if len(available_months) < 2:
                st.warning("Need at least 2 months of data to compare.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    month_a = st.selectbox("Month A", available_months,
                                           index=max(0, len(available_months) - 2), key="comp_a")
                with c2:
                    month_b = st.selectbox("Month B", available_months,
                                           index=len(available_months) - 1, key="comp_b")

                df_a = comp_df[comp_df['year_month'] == month_a]
                df_b = comp_df[comp_df['year_month'] == month_b]

                # --- Side-by-side KPIs ---
                st.markdown("### 📊 Key Metrics")
                k1, k2, k3, k4 = st.columns(4)

                avg_a = df_a['numeric_price'].mean()
                avg_b = df_b['numeric_price'].mean()
                k1.metric(f"Avg Price ({month_a})", f"RM {avg_a:.2f}")
                k1.metric(f"Avg Price ({month_b})", f"RM {avg_b:.2f}", f"{avg_b - avg_a:+.2f}")

                count_a, count_b = len(df_a), len(df_b)
                k2.metric(f"Total Meals ({month_a})", f"{count_a}")
                k2.metric(f"Total Meals ({month_b})", f"{count_b}", f"{count_b - count_a:+d}")

                spend_a = df_a['numeric_price'].sum()
                spend_b = df_b['numeric_price'].sum()
                k3.metric(f"Total Spend ({month_a})", f"RM {spend_a:.2f}")
                k3.metric(f"Total Spend ({month_b})", f"RM {spend_b:.2f}", f"RM {spend_b - spend_a:+.2f}")

                rest_a = df_a['restaurant_name'].nunique()
                rest_b = df_b['restaurant_name'].nunique()
                k4.metric(f"Restaurants ({month_a})", f"{rest_a}")
                k4.metric(f"Restaurants ({month_b})", f"{rest_b}", f"{rest_b - rest_a:+d}")

                st.markdown("---")

                # --- Overlaid Histograms ---
                st.subheader("📈 Price Distribution Comparison")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Histogram(
                    x=df_a['numeric_price'], name=month_a, opacity=0.6, nbinsx=15
                ))
                fig_comp.add_trace(go.Histogram(
                    x=df_b['numeric_price'], name=month_b, opacity=0.6, nbinsx=15
                ))
                fig_comp.update_layout(
                    barmode='overlay', xaxis_title='Price (RM)',
                    yaxis_title='Count', height=400
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # --- Top Restaurants Comparison ---
                st.subheader("🏆 Top Restaurants Comparison")
                cr1, cr2 = st.columns(2)
                with cr1:
                    st.markdown(f"**{month_a}**")
                    rest_a_counts = df_a['restaurant_name'].value_counts().head(5).reset_index()
                    rest_a_counts.columns = ['Restaurant', 'Visits']
                    st.dataframe(rest_a_counts, use_container_width=True, hide_index=True)
                with cr2:
                    st.markdown(f"**{month_b}**")
                    rest_b_counts = df_b['restaurant_name'].value_counts().head(5).reset_index()
                    rest_b_counts.columns = ['Restaurant', 'Visits']
                    st.dataframe(rest_b_counts, use_container_width=True, hide_index=True)

                # --- Meal Type Comparison ---
                st.subheader("🍽️ Meal Type Comparison")
                cm1, cm2 = st.columns(2)
                with cm1:
                    mt_a = df_a['meal_type'].value_counts()
                    fig_ma = px.pie(values=mt_a.values, names=mt_a.index,
                                    title=month_a, hole=0.4)
                    st.plotly_chart(fig_ma, use_container_width=True)
                with cm2:
                    mt_b = df_b['meal_type'].value_counts()
                    fig_mb = px.pie(values=mt_b.values, names=mt_b.index,
                                    title=month_b, hole=0.4)
                    st.plotly_chart(fig_mb, use_container_width=True)
        else:
            st.info("Not enough data for comparison. Adjust filters or scrape more data.")

    # ──────────────────────────────────────────────────────
    # TAB — About & Data Dictionary
    # ──────────────────────────────────────────────────────
    with tab_about:
        st.header("ℹ️ About This Project")

        st.markdown("""
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
        """)

        st.markdown("---")

        st.subheader("📖 Data Dictionary")
        st.markdown("""
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
        """)

        st.markdown("---")
        st.caption(
            "Built with ❤️ using Streamlit · "
            "[Source Code](https://github.com/lordjunn/Foodie-Streamlit) · "
            "[Live Demo](https://junn-foodie-data-science.streamlit.app/)"
        )


else:
    st.info("👈 Start by scraping some data first!")
