from pathlib import Path

import pandas as pd
import numpy as np
import re
from dateutil import parser
from datetime import date, datetime

CSV_PATH = Path(__file__).resolve().parent / "food.csv"
IMAGE_BASE_URL = "https://lordjunn.github.io/Food-MMU/Logs/"
MONTH_TO_NUM = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

STANDARD_COLUMNS = [
    "date",
    "dish_name",
    "restaurant_name",
    "price",
    "numeric_price",
    "meal_type",
    "description",
    "image_url",
]


def _parse_price(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan

    text_clean = str(text).strip().lower().replace(",", "")
    if not text_clean or text_clean == "no price":
        return np.nan

    match = re.search(r"(\d+(?:\.\d+)?)", text_clean)
    if match:
        return float(match.group(1))

    if "free" in text_clean:
        return 0.0
    return np.nan


def _parse_date(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    try:
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return pd.to_datetime(value)
        text_value = str(value).strip()
        if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", text_value):
            return parser.parse(text_value, yearfirst=True, dayfirst=False)
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", text_value):
            return parser.parse(text_value, dayfirst=False)
        return parser.parse(text_value, dayfirst=True, fuzzy=True)
    except Exception:
        return pd.NaT


def _clean_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return str(value).strip()


def _clean_meal_type(value):
    cleaned = _clean_text(value)
    if pd.isna(cleaned):
        return np.nan
    cleaned = re.sub(r"^\((.*)\)$", r"\1", cleaned).strip()
    return cleaned


def _normalize_image_url(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    image_url = str(value).strip()
    if not image_url:
        return np.nan

    if image_url.startswith(("http://", "https://")):
        return image_url

    return IMAGE_BASE_URL + image_url.lstrip('/').replace('\\', '/')


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    out = df.copy()

    if "image" in out.columns and "image_url" not in out.columns:
        out = out.rename(columns={"image": "image_url"})

    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["dish_name"] = out["dish_name"].apply(_clean_text)
    out["restaurant_name"] = out["restaurant_name"].apply(_clean_text)
    out["price"] = out["price"].apply(_clean_text)
    out["meal_type"] = out["meal_type"].apply(_clean_meal_type)
    out["description"] = out["description"].apply(_clean_text)

    out["date"] = out["date"].apply(_parse_date)

    if "numeric_price" in out.columns:
        out["numeric_price"] = pd.to_numeric(out["numeric_price"], errors="coerce")
    out["numeric_price"] = out["numeric_price"].fillna(out["price"].apply(_parse_price))
    out["image_url"] = out["image_url"].apply(_normalize_image_url)

    out = out[STANDARD_COLUMNS]
    out = out.drop_duplicates()
    out = out.sort_values("date", ascending=True, na_position="last")
    return out


def init_db(csv_path: Path = CSV_PATH):
    if csv_path.exists():
        return
    pd.DataFrame(columns=STANDARD_COLUMNS).to_csv(csv_path, index=False)


def load_data_from_db(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    try:
        raw = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return _normalize_df(raw)


def save_data_to_db(df: pd.DataFrame, csv_path: Path = CSV_PATH):
    if df is None or df.empty:
        return

    to_save = _normalize_df(df)
    to_save.to_csv(csv_path, index=False)


def merge_with_existing_and_save(new_df: pd.DataFrame, csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return load_data_from_db(csv_path)

    existing = load_data_from_db(csv_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = _normalize_df(combined)

    save_data_to_db(combined, csv_path)
    return combined


def get_latest_date_from_df(df: pd.DataFrame):
    if df is None or df.empty or "date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.dropna().empty:
        return None
    return parsed.max()


def get_earliest_date_from_df(df: pd.DataFrame):
    if df is None or df.empty or "date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.dropna().empty:
        return None
    return parsed.min()


def build_year_month_pairs(years, months):
    pairs = [(int(y), m) for y in years for m in months if m in MONTH_TO_NUM]
    return sorted(pairs, key=lambda x: (x[0], MONTH_TO_NUM[x[1]]))


def filter_incremental_pairs(year_month_pairs, earliest_date=None, latest_date=None):
    if earliest_date is None and latest_date is None:
        return year_month_pairs

    earliest_key = None
    latest_key = None
    if earliest_date is not None:
        earliest_key = (int(earliest_date.year) % 100, int(earliest_date.month))
    if latest_date is not None:
        latest_key = (int(latest_date.year) % 100, int(latest_date.month))

    incremental = []
    for year, month_name in year_month_pairs:
        month_num = MONTH_TO_NUM.get(month_name)
        if month_num is None:
            continue

        pair_key = (year, month_num)
        should_scrape = False

        # If cached data starts mid-history, backfill months strictly before the earliest month.
        if earliest_key is not None and pair_key < earliest_key:
            should_scrape = True

        # Keep current behavior for new trailing data from the latest cached month onward.
        if latest_key is not None and pair_key >= latest_key:
            should_scrape = True

        if should_scrape:
            incremental.append((year, month_name))
    return incremental
