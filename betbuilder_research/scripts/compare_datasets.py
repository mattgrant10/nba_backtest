#!/usr/bin/env python3
"""
Compare two NBA PlayerStatistics datasets and report key differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tabulate import tabulate


def _safe_read_csv(path: Path, date_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, parse_dates=[date_col], low_memory=False)


def _date_range(df: pd.DataFrame, date_col: str) -> Tuple[Optional[str], Optional[str]]:
    if date_col not in df.columns:
        return None, None
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return str(dates.min().date()), str(dates.max().date())


def _unique_count(df: pd.DataFrame, candidates: List[str]) -> Optional[int]:
    for col in candidates:
        if col in df.columns:
            return int(df[col].nunique())
    return None


def _summary_table(
    name: str,
    df: pd.DataFrame,
    date_col: str
) -> Dict[str, Optional[str]]:
    start, end = _date_range(df, date_col)
    player_count = _unique_count(df, ["personId", "player_id"])
    game_count = _unique_count(df, ["gameId", "game_id"])
    return {
        "dataset": name,
        "rows": f"{len(df):,}",
        "columns": f"{len(df.columns):,}",
        "date_start": start or "N/A",
        "date_end": end or "N/A",
        "unique_players": f"{player_count:,}" if player_count is not None else "N/A",
        "unique_games": f"{game_count:,}" if game_count is not None else "N/A",
    }


def _column_differences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)
    return only_a, only_b


def _dtype_changes(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    shared = sorted(set(df_a.columns) & set(df_b.columns))
    changes = []
    for col in shared:
        a = str(df_a[col].dtype)
        b = str(df_b[col].dtype)
        if a != b:
            changes.append({"column": col, "current_dtype": a, "previous_dtype": b})
    return pd.DataFrame(changes)


def _missingness_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    shared = sorted(set(df_a.columns) & set(df_b.columns))
    if not shared:
        return pd.DataFrame()
    a_missing = df_a[shared].isnull().mean() * 100
    b_missing = df_b[shared].isnull().mean() * 100
    diff = (a_missing - b_missing).abs().sort_values(ascending=False).head(top_n)
    result = pd.DataFrame({
        "column": diff.index,
        "current_missing_pct": a_missing[diff.index].round(2),
        "previous_missing_pct": b_missing[diff.index].round(2),
        "abs_diff_pct": diff.round(2),
    })
    return result.reset_index(drop=True)


def compare_datasets(
    current_path: Path,
    previous_path: Path,
    date_col: str
) -> None:
    current_df = _safe_read_csv(current_path, date_col)
    previous_df = _safe_read_csv(previous_path, date_col)

    summary = [
        _summary_table("current", current_df, date_col),
        _summary_table("previous", previous_df, date_col),
    ]

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(tabulate(summary, headers="keys", tablefmt="grid"))

    only_current, only_previous = _column_differences(current_df, previous_df)
    print("\n" + "=" * 80)
    print("COLUMN DIFFERENCES")
    print("=" * 80)
    print(f"Only in current ({len(only_current)}): {', '.join(only_current) if only_current else 'None'}")
    print(f"Only in previous ({len(only_previous)}): {', '.join(only_previous) if only_previous else 'None'}")

    dtype_changes = _dtype_changes(current_df, previous_df)
    print("\n" + "=" * 80)
    print("DTYPE CHANGES (SHARED COLUMNS)")
    print("=" * 80)
    if dtype_changes.empty:
        print("No dtype differences found.")
    else:
        print(tabulate(dtype_changes, headers="keys", tablefmt="grid", showindex=False))

    missing_diff = _missingness_diff(current_df, previous_df)
    print("\n" + "=" * 80)
    print("MISSINGNESS DIFFERENCES (TOP COLUMNS)")
    print("=" * 80)
    if missing_diff.empty:
        print("No shared columns to compare missingness.")
    else:
        print(tabulate(missing_diff, headers="keys", tablefmt="grid", showindex=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PlayerStatistics datasets.")
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current dataset CSV"
    )
    parser.add_argument(
        "--previous",
        type=Path,
        required=True,
        help="Path to previous dataset CSV"
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="gameDateTimeEst",
        help="Date column to compute timeline (default: gameDateTimeEst)"
    )
    args = parser.parse_args()

    compare_datasets(args.current, args.previous, args.date_col)


if __name__ == "__main__":
    main()
