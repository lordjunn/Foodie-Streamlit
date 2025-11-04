import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# ---------- CONFIG ----------
st.set_page_config(page_title="🍜 MMU Food Log Scraper & Data Explorer", layout="wide")

# ---------- SCRAPER UTILS ----------
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
    """Extract numeric price from string"""
    if not text or text == 'No price':
        return np.nan
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    return float(match.group(1)) if match else np.nan

def scrape_data(years, months):
    base_url = "https://lordjunn.github.io/Food-MMU/Logs/"
    urls = [f"{base_url}{month} {year}.html" for year in years for month in months]
    menu_items = []

    progress = st.progress(0)
    total = len(urls)
    count = 0

    for url in urls:
        count += 1
        progress.progress(count/total)
        try:
            response = requests.get(url)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            for menu_div in soup.find_all('div', class_='menu'):
                date_heading = menu_div.find('h2', class_='menu-group-heading')
                if not date_heading:
                    continue
                date = date_heading.text.strip()
                if 'spendings' in date.lower():
                    continue

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
                    image = item.find('img', class_='menu-item-image')
                    image_url = image.get('src') if image and image.get('src') else 'No image'
                    menu_items.append({
                        'date': date,
                        'dish_name': dish_name,
                        'restaurant_name': restaurant_name,
                        'price': price.text.strip() if price else 'No price',
                        'numeric_price': parse_price(price.text.strip() if price else None),
                        'meal_type': meal_type.text.strip() if meal_type else 'No meal type',
                        'description': preserve_inline_html(description),
                        'image': image_url
                    })
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")
    progress.empty()
    return pd.DataFrame(menu_items)

# ---------- MAIN APP ----------
st.title("🍜 MMU Food Log Scraper & Data Science Explorer")

st.sidebar.header("Scrape Settings")
years = st.sidebar.multiselect("Years", [22, 23, 24, 25], default=[25])
months = st.sidebar.multiselect("Months", 
                                ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
                                default=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov'])
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

# ---------- DATA EXPLORER ----------
st.header("📋 Data Explorer")

if 'data' in st.session_state:
    df = st.session_state['data']

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        rest_filter = st.selectbox("Filter by Restaurant", ["All"] + sorted(df['restaurant_name'].unique().tolist()))
    with col2:
        meal_filter = st.selectbox("Filter by Meal Type", ["All"] + sorted(df['meal_type'].unique().tolist()))
    with col3:
        search = st.text_input("Search Dish Name")

    filtered_df = df.copy()
    if rest_filter != "All":
        filtered_df = filtered_df[filtered_df['restaurant_name'] == rest_filter]
    if meal_filter != "All":
        filtered_df = filtered_df[filtered_df['meal_type'] == meal_filter]
    if search:
        filtered_df = filtered_df[filtered_df['dish_name'].str.contains(search, case=False, na=False)]

    st.dataframe(filtered_df)

    # ---------- STATISTICS ----------
    st.subheader("📊 Quantitative Summary (Price Statistics)")
    if 'numeric_price' in filtered_df.columns:
        numeric_data = filtered_df['numeric_price'].dropna()
        if not numeric_data.empty:
            stats_df = pd.DataFrame({
                'Mean': [numeric_data.mean()],
                'Median': [numeric_data.median()],
                'Min': [numeric_data.min()],
                'Max': [numeric_data.max()],
                'Variance': [numeric_data.var()],
                'Std Dev': [numeric_data.std()],
                'Count': [numeric_data.count()]
            })
            st.dataframe(stats_df)
        else:
            st.info("No numeric price data available.")

    # ---------- CHARTS ----------
    st.header("📈 Visualizations")

    if not filtered_df.empty and 'numeric_price' in filtered_df.columns:
        colA, colB = st.columns(2)

        # Box Plot
        with colA:
            st.subheader("Box Plot: Price by Meal Type")
            fig, ax = plt.subplots()
            sns.boxplot(x='meal_type', y='numeric_price', data=filtered_df, ax=ax, palette="pastel")
            ax.set_xlabel("Meal Type")
            ax.set_ylabel("Price")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # QQ Plot
        with colB:
            st.subheader("QQ Plot (Normality Check)")
            fig2 = plt.figure()
            stats.probplot(filtered_df['numeric_price'].dropna(), dist="norm", plot=plt)
            st.pyplot(fig2)

        # Scatterplot with regression
        st.subheader("Scatter Plot with Regression Line")
        df_time = filtered_df.copy()
        df_time['date'] = pd.to_datetime(df_time['date'], errors='coerce')
        df_time = df_time.dropna(subset=['date', 'numeric_price'])
        if not df_time.empty:
            fig3, ax3 = plt.subplots()
            sns.regplot(x=df_time['date'].map(pd.Timestamp.toordinal),
                        y='numeric_price', data=df_time,
                        scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax3)
            ax3.set_xlabel("Date (ordinal)")
            ax3.set_ylabel("Price")
            st.pyplot(fig3)
        else:
            st.info("Not enough valid date/price data for scatter plot.")
    else:
        st.info("No numeric price data to visualize — try different filters.")
else:
    st.info("👈 Start by scraping some data first!")
