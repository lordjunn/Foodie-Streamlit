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

# ---------- CONFIG ----------
st.set_page_config(page_title="🍜 MMU Food Log Scraper & Data Explorer", layout="wide")
sns.set_theme(style="whitegrid")

# ---------- UTILS ----------
def split_dish_and_restaurant(menu_item_name):
    match = re.match(r"([^\[]+)(?:\s*\[\s*([^\]]+)\s*\])?", menu_item_name.strip())
    if match:
        dish_name = match.group(1).strip()
        restaurant_name = match.group(2).strip() if match.group(2) else 'No restaurant name'
        return dish_name, restaurant_name
    return menu_item_name, 'No restaurant name'

def preserve_inline_html(element):
    if not element:
        return 'No description'
    html = element.decode_contents()
    html = re.sub(r'(?<!<br>)<br\s*/?>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'(<br\s*/?>\s*){2}(?!<br)', '<br>', html, flags=re.IGNORECASE)
    soup_fragment = BeautifulSoup(html, 'html.parser')
    for tag in soup_fragment.find_all(True):
        if tag.name not in ['b', 'i', 'strong', 'em', 'u', 'br', 'small']:
            tag.unwrap()
    return str(soup_fragment).strip()

def parse_price(text):
    if not text or text == 'No price':
        return np.nan
    text = text.replace(',', '')  # Remove commas
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    return float(match.group(1)) if match else np.nan

def normalize_meal_type(x):
    x_lower = x.lower()
    if "breakfast" in x_lower:
        return "Breakfast"
    elif "lunch" in x_lower:
        return "Lunch"
    elif "dinner" in x_lower:
        return "Dinner"
    else:
        return "Other"

# ---------- SCRAPER ----------
def scrape_data(years, months):
    base_url = "https://lordjunn.github.io/Food-MMU/Logs/"
    urls = [f"{base_url}{month} {year}.html" for year in years for month in months]
    menu_items = []

    progress = st.progress(0)
    total = len(urls)
    count = 0

    for url in urls:
        count += 1
        progress.progress(count / total)
        try:
            response = requests.get(url)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')

            for menu_div in soup.find_all('div', class_='menu'):
                # Skip footer-only menu divs
                if menu_div.find('div', id='footer-container'):
                    continue

                # --- Parse the date ---
                date_heading = menu_div.find('h2', class_='menu-group-heading')
                date_obj = None
                if date_heading:
                    date_str = date_heading.text.strip()
                    # remove extra text in parentheses, e.g., "03 Nov 2025 (Monday)"
                    date_str = re.sub(r'\s*\(.*?\)', '', date_str)
                try:
                    date_obj = parser.parse(date_str, dayfirst=True)
                except Exception as e:
                    st.warning(f"Failed to parse date '{date_str}' in {url}: {e}")
                    
                # --- Parse menu items ---
                menu_group = menu_div.find('div', class_='menu-group')
                if not menu_group:
                    continue

                for item in menu_group.find_all('div', class_='menu-item'):
                    name = item.find('span', class_='menu-item-name')
                    if not name or not name.text.strip():
                        continue
                    dish_name, restaurant_name = split_dish_and_restaurant(name.text)

                    price = item.find('span', class_='menu-item-price')
                    meal_type = item.find('span', class_='meal-type')
                    description = item.find('p', class_='menu-item-description')

                    menu_items.append({
                        'date': date_obj,
                        'dish_name': dish_name,
                        'restaurant_name': restaurant_name,
                        'price': price.text.strip() if price else 'No price',
                        'numeric_price': parse_price(price.text.strip() if price else None),
                        'meal_type': normalize_meal_type(meal_type.text.strip()) if meal_type else 'No meal type',
                        'description': preserve_inline_html(description),
                    })

        except Exception as e:
            st.error(f"Error scraping {url}: {e}")

    progress.empty()
    st.success("Scraping completed!")
    return pd.DataFrame(menu_items)


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
years = st.sidebar.multiselect("Years", [22, 23, 24, 25], default=[25])
months = st.sidebar.multiselect(
    "Months",
    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    default=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']
)
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
    df = st.session_state['data']

    # --- Filters ---
    col1, col2, col3 = st.columns(3)
    with col1:
        rest_filter = st.multiselect("Filter by Restaurant", sorted(df['restaurant_name'].unique().tolist()))
    with col2:
        meal_filter = st.multiselect("Filter by Meal Type", sorted(df['meal_type'].unique().tolist()))
    with col3:
        search = st.text_input("Search Dish Name")

    filtered_df = filter_data(df, rest_filter, meal_filter, search)
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

else:
    st.info("👈 Start by scraping some data first!")
