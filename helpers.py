import re
import numpy as np
from bs4 import BeautifulSoup


def split_dish_and_restaurant(menu_item_name):
    match = re.match(r"([^\[]+)(?:\s*\[\s*([^\]]+)\s*\])?", menu_item_name.strip())
    if match:
        dish_name = match.group(1).strip()
        restaurant_name = match.group(2).strip() if match.group(2) else 'No restaurant name'
        return dish_name, restaurant_name
    return menu_item_name.strip(), 'No restaurant name'


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
    result = str(soup_fragment).strip()
    # Strip leading whitespace from each line so Markdown doesn't
    # render indented text as code blocks (4+ spaces = <pre>)
    result = '\n'.join(line.strip() for line in result.splitlines())
    return result


def parse_price(text):
    """
    Extract numeric price for data analysis.
    - If a number exists, return that number.
    - If only 'Free' is mentioned, return 0.0.
    - If nothing valid, return NaN.
    """
    if text is None:
        return np.nan

    text_clean = str(text).strip().lower().replace(',', '')
    if not text_clean or text_clean == 'no price':
        return np.nan

    # If there is a number anywhere (even struck out), use that
    match = re.search(r'(\d+(?:\.\d+)?)', text_clean)
    if match:
        return float(match.group(1))

    # If no number but mentions "free"
    if "free" in text_clean:
        return 0.0

    # Otherwise missing
    return np.nan


def normalize_meal_type(x):
    if not x:
        return "Other"
    x_lower = str(x).lower()
    if "breakfast" in x_lower:
        return "Breakfast"
    elif "lunch" in x_lower:
        return "Lunch"
    elif "dinner" in x_lower:
        return "Dinner"
    else:
        return "Other"
