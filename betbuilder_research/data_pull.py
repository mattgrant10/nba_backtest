from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

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

    raise KeyError(f"Could not find a date column. Available columns:\n{cols}")


def filter_by_year_range(
    year_start: int, year_end: int, input_path: Path, output_path: Path
) -> None:
    if year_end < year_start:
        raise ValueError("year_end must be >= year_start")
    if output_path.exists():
        raise FileExistsError(output_path)
    print(f"[INFO] Loading Excel: {input_path}")
    df = pd.read_excel(input_path, engine="openpyxl")
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
    filtered.to_csv(output_path, index=False)
    print(f"[DONE] wrote {len(filtered):,} rows to {output_path}")


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(
            description="Optional Excel calendar-year extraction; not required for the CSV workflow"
        )
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--year-start", type=int, required=True)
        parser.add_argument("--year-end", type=int, required=True)
        args = parser.parse_args()
        filter_by_year_range(args.year_start, args.year_end, args.input, args.output)
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        raise
