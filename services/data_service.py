import streamlit as st

from db import (
    load_data_from_db,
    save_data_to_db,
    merge_with_existing_and_save,
    get_earliest_date_from_df,
    get_latest_date_from_df,
    build_year_month_pairs,
    filter_incremental_pairs,
)
from scraper import scrape_data_raw, scrape_data_raw_pairs


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def apply_default_image(df, default_img):
    if df is None or df.empty:
        return df
    out = df.copy()
    if "image_url" not in out.columns:
        out["image_url"] = default_img
    out["image_url"] = out["image_url"].fillna("")
    out.loc[out["image_url"].astype(str).str.strip() == "", "image_url"] = default_img
    return out


def load_or_bootstrap_data(default_years, default_months, default_img):
    db_df = apply_default_image(load_data_from_db(), default_img)
    if not db_df.empty:
        return db_df

    scraped = apply_default_image(scrape_data_raw(default_years, default_months), default_img)
    if not scraped.empty:
        save_data_to_db(scraped)
    return scraped


def run_incremental_scrape(requested_years, requested_months, default_img):
    existing_df = apply_default_image(load_data_from_db(), default_img)
    earliest_date = get_earliest_date_from_df(existing_df)
    latest_date = get_latest_date_from_df(existing_df)
    all_pairs = build_year_month_pairs(requested_years, requested_months)

    if earliest_date is None and latest_date is None:
        scrape_pairs = all_pairs
    else:
        scrape_pairs = filter_incremental_pairs(
            all_pairs,
            earliest_date=earliest_date,
            latest_date=latest_date,
        )

    if not scrape_pairs:
        st.info("CSV cache is already up to date for your selected months.")
        return existing_df

    with st.spinner(f"Scraping {len(scrape_pairs)} month(s) not fully cached in CSV..."):
        scraped_df = apply_default_image(scrape_data_raw_pairs(scrape_pairs), default_img)

    if scraped_df.empty:
        st.warning("Scraping finished, but no new rows were found.")
        return existing_df

    merged_df = apply_default_image(merge_with_existing_and_save(scraped_df), default_img)
    st.success(f"✅ Added {len(scraped_df)} scraped rows. CSV cache now has {len(merged_df)} rows.")
    return merged_df


def filter_data(df, restaurants, meal_types, search):
    filtered_df = df.copy()
    if restaurants:
        filtered_df = filtered_df[filtered_df["restaurant_name"].isin(restaurants)]
    if meal_types:
        filtered_df = filtered_df[filtered_df["meal_type"].isin(meal_types)]
    if search:
        filtered_df = filtered_df[
            filtered_df["dish_name"].str.contains(search, case=False, na=False)
        ]
    return filtered_df


def handle_scrape(df):
    if not df.empty:
        st.success(f"✅ Loaded {len(df)} items!")
        st.session_state["data"] = df
    else:
        st.warning("No data found.")
