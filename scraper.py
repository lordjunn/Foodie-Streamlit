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

# ---------- UTILS ----------
from helpers import (
    split_dish_and_restaurant,
    preserve_inline_html,
    parse_price,
    normalize_meal_type
)

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
                    
                    # Only parse if it looks like a proper date
                    if re.match(r'\d{1,2}\s+\w+\s+\d{4}', date_str):
                        # remove extra text in parentheses, e.g., "03 Nov 2025 (Monday)"
                        date_str = re.sub(r'\s*\(.*?\)', '', date_str)
                        try:
                            date_obj = parser.parse(date_str, dayfirst=True)
                        except Exception as e:
                            st.warning(f"Failed to parse date '{date_str}' in {url}: {e}")
                    else:
                        continue  # skip this menu_div entirely

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