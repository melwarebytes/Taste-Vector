"""Data loading utilities for TasteVector."""
"""
data_loader.py — Data Ingestion & Validation
============================================
Person 1 | TasteVector Project

Single point of contact with raw CSV files.
Reads restaurants.csv, users.csv, ratings.csv into clean Pandas DataFrames.
All other modules receive DataFrames or NumPy arrays — never raw CSV paths.
"""

import os
import pandas as pd


# ── Expected schema ──────────────────────────────────────────────────────────

RESTAURANT_COLS = {"restaurant_id", "name", "cuisine", "price",
                   "spice", "distance_km", "veg_friendly"}

USER_COLS = {"user_id", "name", "preferred_cuisine",
             "max_price", "spice_tolerance", "max_distance"}

RATING_COLS = {"user_id", "restaurant_id", "rating"}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_restaurants(data_dir: str) -> pd.DataFrame:
    """
    Load and validate restaurants.csv.

    Columns: restaurant_id, name, cuisine, price (1-5), spice (1-5),
             distance_km (float), veg_friendly (0 or 1)
    """
    path = os.path.join(data_dir, "restaurants.csv")
    df = pd.read_csv(path)

    _check_columns(df, RESTAURANT_COLS, "restaurants.csv")

    # Type coercions
    df["restaurant_id"] = pd.to_numeric(df["restaurant_id"], errors="coerce")
    df["price"]         = pd.to_numeric(df["price"],         errors="coerce")
    df["spice"]         = pd.to_numeric(df["spice"],         errors="coerce")
    df["distance_km"]   = pd.to_numeric(df["distance_km"],   errors="coerce")
    df["veg_friendly"]  = pd.to_numeric(df["veg_friendly"],  errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(RESTAURANT_COLS))
    _warn_dropped(before, len(df), "restaurants.csv")

    df = df.reset_index(drop=True)
    return df


def load_users(data_dir: str) -> pd.DataFrame:
    """
    Load and validate users.csv.

    Columns: user_id, name, preferred_cuisine, max_price (1-5),
             spice_tolerance (1-5), max_distance (float)
    """
    path = os.path.join(data_dir, "users.csv")
    df = pd.read_csv(path)

    _check_columns(df, USER_COLS, "users.csv")

    df["user_id"]         = pd.to_numeric(df["user_id"],         errors="coerce")
    df["max_price"]       = pd.to_numeric(df["max_price"],       errors="coerce")
    df["spice_tolerance"] = pd.to_numeric(df["spice_tolerance"], errors="coerce")
    df["max_distance"]    = pd.to_numeric(df["max_distance"],    errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(USER_COLS))
    _warn_dropped(before, len(df), "users.csv")

    df = df.reset_index(drop=True)
    return df


def load_ratings(data_dir: str) -> pd.DataFrame:
    """
    Load and validate ratings.csv.

    Columns: user_id, restaurant_id, rating (1.0 - 5.0)
    """
    path = os.path.join(data_dir, "ratings.csv")
    df = pd.read_csv(path)

    _check_columns(df, RATING_COLS, "ratings.csv")

    df["user_id"]       = pd.to_numeric(df["user_id"],       errors="coerce")
    df["restaurant_id"] = pd.to_numeric(df["restaurant_id"], errors="coerce")
    df["rating"]        = pd.to_numeric(df["rating"],        errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(RATING_COLS))
    # Clamp ratings to valid range
    df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]
    _warn_dropped(before, len(df), "ratings.csv")

    df = df.reset_index(drop=True)
    return df


def load_all(data_dir: str) -> tuple:
    """
    Load all three CSVs at once.

    Returns
    -------
    (restaurants, users, ratings) as clean DataFrames
    """
    restaurants = load_restaurants(data_dir)
    users       = load_users(data_dir)
    ratings     = load_ratings(data_dir)
    return restaurants, users, ratings


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def _warn_dropped(before: int, after: int, filename: str) -> None:
    dropped = before - after
    if dropped > 0:
        print(f"[data_loader] WARNING: dropped {dropped} malformed "
              f"row(s) from {filename}")


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    restaurants, users, ratings = load_all(os.path.join(base, "data"))

    print(f"Restaurants : {len(restaurants)} rows")
    print(restaurants.head(3).to_string(index=False))
    print(f"\nUsers       : {len(users)} rows")
    print(users.head(3).to_string(index=False))
    print(f"\nRatings     : {len(ratings)} rows")
    print(ratings.head(5).to_string(index=False))

