import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "food_log_cache.db"
TABLE_NAME = "menu_items"
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


def init_db(db_path: Path = DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                date TEXT,
                dish_name TEXT,
                restaurant_name TEXT,
                price TEXT,
                numeric_price REAL,
                meal_type TEXT,
                description TEXT,
                image_url TEXT
            )
            """
        )


def load_data_from_db(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def save_data_to_db(df: pd.DataFrame, db_path: Path = DB_PATH):
    if df is None or df.empty:
        return

    to_save = df.copy()
    if "date" in to_save.columns:
        to_save["date"] = pd.to_datetime(to_save["date"], errors="coerce")

    with sqlite3.connect(db_path) as conn:
        to_save.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)


def merge_with_existing_and_save(new_df: pd.DataFrame, db_path: Path = DB_PATH) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return load_data_from_db(db_path)

    existing = load_data_from_db(db_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates()

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.sort_values("date", ascending=True, na_position="last")

    save_data_to_db(combined, db_path)
    return combined


def get_latest_date_from_df(df: pd.DataFrame):
    if df is None or df.empty or "date" not in df.columns:
        return None
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.dropna().empty:
        return None
    return parsed.max()


def build_year_month_pairs(years, months):
    pairs = [(int(y), m) for y in years for m in months if m in MONTH_TO_NUM]
    return sorted(pairs, key=lambda x: (x[0], MONTH_TO_NUM[x[1]]))


def filter_incremental_pairs(year_month_pairs, latest_date):
    if latest_date is None:
        return year_month_pairs

    latest_key = (int(latest_date.year) % 100, int(latest_date.month))
    incremental = []
    for year, month_name in year_month_pairs:
        month_num = MONTH_TO_NUM.get(month_name)
        if month_num is None:
            continue
        if (year, month_num) >= latest_key:
            incremental.append((year, month_name))
    return incremental
