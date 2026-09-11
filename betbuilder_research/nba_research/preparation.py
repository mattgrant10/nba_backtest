"""Preserved research calculations: preparation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate
from src.config import get_logger
from src.visualization import display_dataframe
from .plotting import init_dataset_context

logger = get_logger(__name__)


def load_and_display_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data and display comprehensive information."""
    logger.info("=" * 100)
    logger.info("STEP 1: LOADING RAW NBA PLAYER STATISTICS DATA")
    logger.info("=" * 100)

    logger.info(f"\n>>> Loading data from: {file_path}")
    logger.info(f">>> File size: {file_path.stat().st_size / 1024**2:.2f} MB")

    # Load data
    df = pd.read_csv(file_path, parse_dates=["gameDateTimeEst"])

    logger.info(f"✓ Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
    init_dataset_context(df, date_col="gameDateTimeEst")

    # Display raw data
    display_dataframe(df, "Raw Player Statistics Data", max_rows=20)

    # Show column types
    logger.info("\n>>> Column Data Types:")
    dtype_df = pd.DataFrame(
        {
            "Column": df.dtypes.index,
            "Type": df.dtypes.values.astype(str),
            "Non-Null": df.count().values,
            "Null": df.isnull().sum().values,
            "Unique": df.nunique().values,
        }
    )
    print("\n" + tabulate(dtype_df, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Memory usage
    logger.info(f"\n>>> Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 2: CLEANING AND PREPARING DATA")
    logger.info("=" * 100)

    df_clean = df.copy()

    # Create player full name
    logger.info("\n>>> Creating player full name...")
    df_clean["player_name"] = df_clean["firstName"] + " " + df_clean["lastName"]

    # Convert date
    logger.info(">>> Processing dates...")
    df_clean["game_date"] = pd.to_datetime(df_clean["gameDateTimeEst"])
    df_clean["game_date_only"] = df_clean["game_date"].dt.date
    df_clean["day_of_week"] = df_clean["game_date"].dt.day_name()
    df_clean["month"] = df_clean["game_date"].dt.month
    df_clean["week"] = df_clean["game_date"].dt.isocalendar().week

    # Create team names
    logger.info(">>> Creating team identifiers...")
    df_clean["player_team"] = df_clean["playerteamCity"] + " " + df_clean["playerteamName"]
    df_clean["opponent_team"] = df_clean["opponentteamCity"] + " " + df_clean["opponentteamName"]

    # Convert win to boolean
    df_clean["win"] = df_clean["win"].astype(bool)
    df_clean["home"] = df_clean["home"].astype(bool)

    # Create meaningful minutes played
    df_clean["minutes_played"] = df_clean["numMinutes"]

    # Rename key stats for clarity
    stat_renames = {
        "points": "PTS",
        "assists": "AST",
        "reboundsTotal": "REB",
        "threePointersMade": "FG3M",
        "steals": "STL",
        "blocks": "BLK",
        "turnovers": "TOV",
        "fieldGoalsMade": "FGM",
        "fieldGoalsAttempted": "FGA",
        "threePointersAttempted": "FG3A",
        "freeThrowsMade": "FTM",
        "freeThrowsAttempted": "FTA",
        "plusMinusPoints": "PLUS_MINUS",
    }

    for old, new in stat_renames.items():
        if old in df_clean.columns:
            df_clean[new] = df_clean[old]

    # Calculate shooting percentages (handle division by zero)
    logger.info(">>> Calculating shooting percentages...")
    df_clean["FG_PCT"] = np.where(df_clean["FGA"] > 0, df_clean["FGM"] / df_clean["FGA"], 0)
    df_clean["FG3_PCT"] = np.where(df_clean["FG3A"] > 0, df_clean["FG3M"] / df_clean["FG3A"], 0)
    df_clean["FT_PCT"] = np.where(df_clean["FTA"] > 0, df_clean["FTM"] / df_clean["FTA"], 0)

    # Calculate per-minute stats
    logger.info(">>> Calculating per-minute statistics...")
    df_clean["PTS_PER_MIN"] = np.where(
        df_clean["minutes_played"] > 0, df_clean["PTS"] / df_clean["minutes_played"], 0
    )
    df_clean["AST_PER_MIN"] = np.where(
        df_clean["minutes_played"] > 0, df_clean["AST"] / df_clean["minutes_played"], 0
    )
    df_clean["REB_PER_MIN"] = np.where(
        df_clean["minutes_played"] > 0, df_clean["REB"] / df_clean["minutes_played"], 0
    )

    # Filter out players with < 5 minutes (likely DNP or garbage time)
    logger.info("\n>>> Filtering players with < 5 minutes played...")
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean["minutes_played"] >= 5].copy()
    filtered_count = initial_count - len(df_clean)
    logger.info(
        f"    Removed {filtered_count:,} rows ({filtered_count / initial_count * 100:.1f}%)"
    )
    logger.info(f"    Remaining: {len(df_clean):,} rows")

    display_dataframe(df_clean, "Cleaned Player Statistics Data", max_rows=15)

    return df_clean
