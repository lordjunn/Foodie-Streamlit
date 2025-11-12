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
    """
    Extract a numeric price from text. 
    Handles cases like:
        "<s>RM 8.50</s>Free??"  →  0.0
        "Free" or "FREE!!"      →  0.0
        "RM 12.00"              →  12.0
    Returns NaN if no valid price info found.
    """
    if not text or text == 'No price':
        return np.nan

    text_clean = text.strip().lower()

    # --- Handle "free" first ---
    if re.search(r'\bfree\b', text_clean):
        return 0.0

    # --- Otherwise, look for a number (possibly with RM or $) ---
    text_clean = text_clean.replace(',', '')
    match = re.search(r'(\d+(?:\.\d+)?)', text_clean)

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