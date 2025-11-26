"""Preprocessing utilities for player game data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import get_logger


logger = get_logger(__name__)


def ensure_categorical(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convert specified columns to categorical dtype for memory efficiency.

    Args:
        df: Input DataFrame
        columns: Column names to convert

    Returns:
        DataFrame with categorical columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
            logger.debug(f"Converted {col} to categorical")
    return df


def add_player_rest_days(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days of rest for each player between games.

    Args:
        player_games: DataFrame with player_id and game_date columns

    Returns:
        DataFrame with added days_rest column
    """
    df = player_games.sort_values(["player_id", "game_date"]).copy()

    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)

    df["days_rest"] = (
        pd.to_datetime(df["game_date"]) - pd.to_datetime(df["prev_game_date"])
    ).dt.days

    # Fill missing values with median rest days
    median_rest = df["days_rest"].median()
    df["days_rest"] = df["days_rest"].fillna(median_rest)

    logger.info(
        f"Added rest days (median: {median_rest:.1f}, "
        f"range: {df['days_rest'].min():.0f}-{df['days_rest'].max():.0f})"
    )

    df.drop(columns=["prev_game_date"], inplace=True)

    return df


def add_basic_context_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic contextual features to player game data.

    Features added:
        - is_home: Binary indicator for home games
        - days_rest: Days since last game for each player

    Args:
        player_games: DataFrame with player game logs

    Returns:
        DataFrame with added context features
    """
    df = player_games.copy()

    # Add home/away indicator
    if "home_away" in df.columns:
        df["is_home"] = (df["home_away"] == "H").astype(int)
        home_pct = df["is_home"].mean() * 100
        logger.info(f"Added is_home feature ({home_pct:.1f}% home games)")
    else:
        df["is_home"] = np.nan
        logger.warning("home_away column not found, is_home set to NaN")

    # Add rest days
    df = add_player_rest_days(df)

    return df


def filter_low_activity_players(
    player_games: pd.DataFrame,
    min_games: int = 20,
    min_minutes_per_game: float = 5.0
) -> pd.DataFrame:
    """
    Filter out players with insufficient game history or playing time.

    Args:
        player_games: DataFrame with player game logs
        min_games: Minimum number of games required
        min_minutes_per_game: Minimum average minutes per game

    Returns:
        Filtered DataFrame
    """
    df = player_games.copy()

    # Calculate games per player
    player_counts = df.groupby("player_id").size()
    valid_players_games = player_counts[player_counts >= min_games].index

    # Calculate average minutes if available
    if "minutes" in df.columns:
        player_minutes = df.groupby("player_id")["minutes"].mean()
        valid_players_minutes = player_minutes[
            player_minutes >= min_minutes_per_game
        ].index

        # Take intersection
        valid_players = set(valid_players_games) & set(valid_players_minutes)

        logger.info(
            f"Filtering: {len(player_counts)} total players -> "
            f"{len(valid_players_games)} with >={min_games} games -> "
            f"{len(valid_players)} with >={min_minutes_per_game:.1f} avg minutes"
        )
    else:
        valid_players = set(valid_players_games)
        logger.info(
            f"Filtering: {len(player_counts)} total players -> "
            f"{len(valid_players)} with >={min_games} games"
        )

    df_filtered = df[df["player_id"].isin(valid_players)].copy()

    logger.info(f"Kept {len(df_filtered):,} / {len(df):,} games")

    return df_filtered


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Handle missing values in specified columns.

    Args:
        df: Input DataFrame
        strategy: Strategy for imputation ('median', 'mean', 'zero', 'drop')
        columns: Columns to process (if None, processes all numeric columns)

    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        if strategy == "median":
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            logger.debug(f"Filled {n_missing} missing values in {col} with median={fill_value:.2f}")

        elif strategy == "mean":
            fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)
            logger.debug(f"Filled {n_missing} missing values in {col} with mean={fill_value:.2f}")

        elif strategy == "zero":
            df[col] = df[col].fillna(0)
            logger.debug(f"Filled {n_missing} missing values in {col} with 0")

        elif strategy == "drop":
            df = df.dropna(subset=[col])
            logger.debug(f"Dropped {n_missing} rows with missing {col}")

        else:
            logger.warning(f"Unknown strategy '{strategy}', skipping {col}")

    return df


def add_rolling_averages(
    player_games: pd.DataFrame,
    columns: list[str],
    windows: list[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Add rolling average features for specified columns.

    Args:
        player_games: DataFrame with player game logs
        columns: Columns to compute rolling averages for
        windows: Window sizes for rolling averages

    Returns:
        DataFrame with added rolling average columns
    """
    df = player_games.sort_values(["player_id", "game_date"]).copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping rolling averages")
            continue

        for window in windows:
            new_col = f"{col}_rolling_{window}"
            df[new_col] = df.groupby("player_id")[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            logger.debug(f"Added {new_col}")

    return df
