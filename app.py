import streamlit as st
import pandas as pd
import seaborn as sns

from db import init_db
from services.data_service import (
    convert_df,
    load_or_bootstrap_data,
    run_incremental_scrape,
    handle_scrape,
)
from ui.filters import render_filters
from ui.metrics import render_kpis
from tabs.dashboard import render_dashboard
from tabs.data_stats import render_data_stats
from tabs.visualizations import render_visualizations
from tabs.forecasting_tab import render_forecasting
from tabs.food_gallery import render_food_gallery
from tabs.compare import render_compare
from tabs.about import render_about


st.set_page_config(page_title="🍜 Junn Food Log Scraper & Data Explorer", layout="wide")
sns.set_theme(style="whitegrid")


TAHUN = [22, 23, 24, 25, 26]
BULAN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_IMG = "https://i.ytimg.com/vi/geOCvzwdt-s/maxresdefault.jpg"
DEFAULT_YEARS = [25]
DEFAULT_MONTHS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


st.title("🍜 Junn Food Log Scraper & Data Science Explorer")

st.sidebar.header("Scrape Settings")
years = st.sidebar.multiselect("Years", TAHUN, default=DEFAULT_YEARS)
months = st.sidebar.multiselect(
    "Months",
    BULAN,
    default=DEFAULT_MONTHS,
)
scrape_button = st.sidebar.button("🔍 Start Scraping")
st.sidebar.markdown("---")
scrape_all_button = st.sidebar.button("🌎 Scrape All (Ignore Filters)")
clear_cache = st.sidebar.button("🗑️ Clear Cache")

if clear_cache:
    st.cache_data.clear()
    st.toast("In-memory cache cleared (CSV cache is kept).")

init_db()

if scrape_button:
    handle_scrape(run_incremental_scrape(years, months, DEFAULT_IMG))

if scrape_all_button:
    handle_scrape(run_incremental_scrape(TAHUN, BULAN, DEFAULT_IMG))

if "data" not in st.session_state:
    auto_df = load_or_bootstrap_data(DEFAULT_YEARS, DEFAULT_MONTHS, DEFAULT_IMG)
    if not auto_df.empty:
        st.session_state["data"] = auto_df

st.header("📋 Data Explorer")
if "data" in st.session_state:
    df = st.session_state["data"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    filtered_df = render_filters(df)
    render_kpis(filtered_df)

    tab_dash, tab_data, tab_viz, tab_forecast, tab_gallery, tab_compare, tab_about = st.tabs(
        [
            "🏠 Dashboard",
            "📋 Data & Stats",
            "📈 Visualizations",
            "🔮 Forecasting",
            "🍽️ Food Images",
            "🔄 Compare",
            "ℹ️ About",
        ]
    )

    with tab_dash:
        render_dashboard(filtered_df)

    with tab_data:
        render_data_stats(filtered_df, convert_df)

    with tab_viz:
        render_visualizations(filtered_df)

    with tab_forecast:
        render_forecasting(filtered_df)

    with tab_gallery:
        render_food_gallery(filtered_df, DEFAULT_IMG)

    with tab_compare:
        render_compare(filtered_df)

    with tab_about:
        render_about()
else:
    st.info("👈 Start by scraping some data first!")
