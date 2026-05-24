import streamlit as st
import pandas as pd

from helpers import normalize_meal_type
import plots


def render_visualizations(filtered_df):
    st.header("📈 Interactive Visualizations")

    if filtered_df is None or filtered_df.empty or "numeric_price" not in filtered_df.columns:
        st.info("Not enough data to visualize. Adjust your filters or scrape more data.")
        return

    vis_df = filtered_df.copy()
    vis_df["meal_category"] = vis_df["meal_type"].apply(normalize_meal_type)
    numeric_prices = vis_df["numeric_price"].dropna()

    if numeric_prices.empty:
        st.info("Not enough data to visualize. Adjust your filters or scrape more data.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Box Plot: Price by Meal Category")
        fig_box = plots.plot_box_price_by_category(vis_df)
        st.plotly_chart(fig_box, width="stretch")

    with col_b:
        st.subheader("QQ Plot (Normality Check)")
        fig_qq = plots.plot_qq_normality_check(numeric_prices)
        st.plotly_chart(fig_qq, width="stretch")

    st.subheader("Price Distribution Histogram")
    fig_hist = plots.plot_price_distribution(numeric_prices)
    st.plotly_chart(fig_hist, width="stretch")

    st.subheader("📅 Prices Over Time")
    vis_df["date"] = pd.to_datetime(vis_df["date"], errors="coerce", dayfirst=True)
    fig_time = plots.plot_prices_over_time(vis_df)

    if fig_time:
        st.plotly_chart(fig_time, width="stretch")
    else:
        st.warning("Not enough data points to plot 'Prices Over Time'.")

    st.subheader("📅 Eating Habits Heatmap")
    fig_cal = plots.plot_calendar_heatmap(vis_df)
    st.plotly_chart(fig_cal, width="stretch")
