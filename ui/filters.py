import streamlit as st
import pandas as pd

from services.data_service import filter_data


def render_filters(df):
    if df is None or df.empty:
        return df

    with st.expander("Filters", expanded=False):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
        with col1:
            rest_filter = st.multiselect(
                "Filter by Restaurant",
                options=sorted(df["restaurant_name"].dropna().unique().tolist()),
            )
        with col2:
            meal_filter = st.multiselect(
                "Filter by Meal Type",
                options=sorted(df["meal_type"].dropna().unique().tolist()),
            )
        with col3:
            search = st.text_input("Search Dish Name")
        with col4:
            min_date = df["date"].min()
            max_date = df["date"].max()
            date_range = st.date_input(
                "Filter by Date",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY",
            )

        filtered_df = filter_data(df, rest_filter, meal_filter, search)

        ignore_zero = st.checkbox(
            "Ignore zero or near-zero prices (numeric_price < 0.01)", value=False
        )
        if ignore_zero:
            filtered_df = filtered_df[filtered_df["numeric_price"] >= 0.01]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0], dayfirst=True)
            end_date = pd.to_datetime(date_range[1], dayfirst=True)
            filtered_df = filtered_df[
                (filtered_df["date"] >= start_date)
                & (filtered_df["date"] <= end_date)
            ]

    return filtered_df
