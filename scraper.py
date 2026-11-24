import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from dateutil import parser
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# ---------- UTILS ----------
from helpers import (
    split_dish_and_restaurant,
    preserve_inline_html,
    parse_price,
    normalize_meal_type,
)

try:
    from config import DEFAULT_IMG
except ImportError:
    DEFAULT_IMG = "https://i.ytimg.com/vi/geOCvzwdt-s/maxresdefault.jpg"

logger = logging.getLogger(__name__)


# ---------- HELPERS ----------
def _build_urls(years, months):
    base_url = "https://lordjunn.github.io/Food-MMU/Logs/"
    return [f"{base_url}{month} {year}.html" for year in years for month in months]


def _build_urls_from_pairs(year_month_pairs):
    base_url = "https://lordjunn.github.io/Food-MMU/Logs/"
    return [f"{base_url}{month} {year}.html" for year, month in year_month_pairs]


def _parse_page(soup, page_url):
    """Parse a single page's BeautifulSoup into a list of item dicts."""
    def _parse_date_heading(date_text):
        cleaned = date_text.strip()
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", cleaned):
            return parser.parse(cleaned, dayfirst=False)
        return parser.parse(cleaned, dayfirst=True)

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
                    date_obj = _parse_date_heading(date_str)
                except Exception:
                    continue
            elif re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", date_str):
                try:
                    date_obj = _parse_date_heading(date_str)
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
            image_url = DEFAULT_IMG
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


def _scrape_single_url(url: str):
    """Fetch and parse a single page URL."""
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            logger.info("URL returned %s: %s", response.status_code, url)
            return []
        soup = BeautifulSoup(response.text, 'html.parser')
        return _parse_page(soup, url)
    except Exception as e:
        logger.warning("Error scraping %s: %s", url, e)
        return []


# ---------- SCRAPER (no Streamlit widgets -- safe for @st.cache_data) ----------
def scrape_data_raw(years, months, max_workers: int = 6):
    """Scrape concurrently without Streamlit UI elements."""
    urls = _build_urls(years, months)
    if not urls:
        return pd.DataFrame()

    menu_items = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for items in executor.map(_scrape_single_url, urls):
            menu_items.extend(items)
    return pd.DataFrame(menu_items)


def scrape_data_raw_pairs(year_month_pairs, max_workers: int = 6):
    """Scrape year-month tuples concurrently."""
    urls = _build_urls_from_pairs(year_month_pairs)
    if not urls:
        return pd.DataFrame()

    menu_items = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for items in executor.map(_scrape_single_url, urls):
            menu_items.extend(items)
    return pd.DataFrame(menu_items)


# ---------- SCRAPER (with Streamlit progress bar) ----------
def scrape_data(years, months, max_workers: int = 6):
    """Scrape concurrently with Streamlit progress bar for manual use."""
    urls = _build_urls(years, months)
    if not urls:
        return pd.DataFrame()

    menu_items = []
    progress = st.progress(0)
    total = len(urls)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scrape_single_url, url): url for url in urls}
        for future in as_completed(futures):
            completed += 1
            progress.progress(completed / total)
            try:
                items = future.result()
                menu_items.extend(items)
            except Exception as e:
                st.error(f"Error scraping {futures[future]}: {e}")

    progress.empty()
    st.success("Scraping completed!")
    return pd.DataFrame(menu_items)
