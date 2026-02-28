import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from dateutil import parser
from urllib.parse import urljoin

# ---------- UTILS ----------
from helpers import (
    split_dish_and_restaurant,
    preserve_inline_html,
    parse_price,
    normalize_meal_type
)


# ---------- HELPERS ----------
def _build_urls(years, months):
    base_url = "https://lordjunn.github.io/Food-MMU/Logs/"
    return [f"{base_url}{month} {year}.html" for year in years for month in months]


def _parse_page(soup, page_url):
    """Parse a single page's BeautifulSoup into a list of item dicts."""
    items = []
    for menu_div in soup.find_all('div', class_='menu'):
        if menu_div.find('div', id='footer-container'):
            continue

        date_heading = menu_div.find('h2', class_='menu-group-heading')
        date_obj = None
        if date_heading:
            date_str = date_heading.text.strip()
            if re.match(r'\d{1,2}\s+\w+\s+\d{4}', date_str):
                date_str = re.sub(r'\s*\(.*?\)', '', date_str)
                try:
                    date_obj = parser.parse(date_str, dayfirst=True)
                except Exception:
                    continue
            else:
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

            # --- Extract food image URL ---
            image_tag = item.find('img', class_='menu-item-image')
            image_url = None
            if image_tag and image_tag.get('src'):
                src = image_tag['src']
                if not src.startswith(('http://', 'https://')):
                    src = urljoin(page_url, src)
                image_url = src

            items.append({
                'date': date_obj,
                'dish_name': dish_name,
                'restaurant_name': restaurant_name,
                'price': price.text.strip() if price else 'No price',
                'numeric_price': parse_price(price.text.strip() if price else None),
                'meal_type': normalize_meal_type(meal_type.text.strip()) if meal_type else 'No meal type',
                'description': preserve_inline_html(description),
                'image_url': image_url,
            })

    return items


# ---------- SCRAPER (no Streamlit widgets -- safe for @st.cache_data) ----------
def scrape_data_raw(years, months):
    """Scrape without Streamlit UI elements. Safe for caching."""
    urls = _build_urls(years, months)
    menu_items = []
    for url in urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            menu_items.extend(_parse_page(soup, url))
        except Exception:
            pass
    return pd.DataFrame(menu_items)


# ---------- SCRAPER (with Streamlit progress bar) ----------
def scrape_data(years, months):
    """Scrape with Streamlit progress bar for manual use."""
    urls = _build_urls(years, months)
    menu_items = []
    progress = st.progress(0)
    total = len(urls)

    for count, url in enumerate(urls, 1):
        progress.progress(count / total)
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            menu_items.extend(_parse_page(soup, url))
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")

    progress.empty()
    st.success("Scraping completed!")
    return pd.DataFrame(menu_items)
