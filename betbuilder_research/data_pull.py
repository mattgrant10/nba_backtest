from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Path to the big Excel master file
INPUT_PATH = Path(
    "/Users/matthewraymondandrewgrant/PycharmProjects/"
    "nba_backtest/betbuilder_research/data/raw/PlayerStatistics1.xlsx"
)

# Year range you want
YEAR_START = 2018
YEAR_END = 2025  # inclusive

# Output CSV
OUTPUT_PATH = Path(
    f"/Users/matthewraymondandrewgrant/PycharmProjects/"
    f"nba_backtest/betbuilder_research/data/raw/PlayerStatistics_{YEAR_START}_{YEAR_END}.csv"
)

# Prefer this name if present
PREFERRED_DATE_COL = "gameDate"


def find_date_column(df: pd.DataFrame) -> str:
    """
    Try to locate the date column in the dataframe.

    Priority:
      1. Exact 'gameDate'
      2. Any column whose name contains 'gamedate'
      3. Any column whose name contains 'date'
    """
    cols = list(df.columns)
    lower_cols = [str(c).lower() for c in cols]

    # 1. Exact match
    if PREFERRED_DATE_COL in cols:
        return PREFERRED_DATE_COL

    # 2. Contains 'gamedate'
    for col, low in zip(cols, lower_cols):
        if "gamedate" in low:
            return col

    # 3. Any 'date'
    for col, low in zip(cols, lower_cols):
        if "date" in low:
            return col

    raise KeyError(
        f"Could not find a date column. Available columns:\n{cols}"
    )


def filter_by_year_range(year_start: int, year_end: int) -> None:
    print(f"[INFO] Loading Excel: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH, engine="openpyxl")
    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns")

    print("[INFO] Columns:", list(df.columns))

    # Find actual date column
    date_col = find_date_column(df)
    print(f"[INFO] Using '{date_col}' as the game date column")

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

    # Build calendar bounds
    start_bound = f"{year_start}-01-01"
    end_bound = f"{year_end + 1}-01-01"  # strictly less than this

    dt = df[date_col]
    mask = (dt >= start_bound) & (dt < end_bound)
    filtered = df.loc[mask].copy()

    print(f"[INFO] Keeping rows with {year_start}–{year_end} dates")
    print(f"[INFO] Kept {len(filtered):,} rows")

    # Save to CSV
    filtered.to_csv(OUTPUT_PATH, index=False)
    print(f"[DONE] wrote {len(filtered):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        filter_by_year_range(YEAR_START, YEAR_END)
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        raise