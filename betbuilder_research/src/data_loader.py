"""Data loading module with validation and error handling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import RAW_DATA_DIR, get_logger


logger = get_logger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


def _read_csv(
    path: Path,
    parse_dates: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read CSV file with error handling and validation.

    Args:
        path: Path to CSV file
        parse_dates: Columns to parse as dates
        required_columns: Columns that must be present

    Returns:
        DataFrame with loaded data

    Raises:
        DataLoadError: If file doesn't exist or required columns are missing
    """
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
    except Exception as e:
        raise DataLoadError(f"Error reading {path}: {str(e)}")

    # Validate required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataLoadError(
                f"Missing required columns in {path.name}: {missing_cols}"
            )

    return df


def load_player_game_logs(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load historic player game logs.

    Expected columns:
        - game_id: Unique game identifier
        - game_date: Date of the game
        - season: Season identifier
        - player_id: Unique player identifier
        - player_name: Player name
        - team: Player's team
        - opponent: Opponent team
        - home_away: 'H' for home, 'A' for away
        - pts: Points scored
        - fg3m: Three-pointers made
        - ast: Assists
        - reb: Rebounds
        - minutes: Minutes played

    Args:
        path: Optional path to CSV file (defaults to RAW_DATA_DIR/player_game_logs.csv)

    Returns:
        DataFrame with player game logs

    Raises:
        DataLoadError: If file doesn't exist or required columns are missing
    """
    if path is None:
        path = RAW_DATA_DIR / "player_game_logs.csv"

    required_columns = [
        "game_id", "game_date", "player_id", "pts", "fg3m", "ast", "reb"
    ]

    df = _read_csv(path, parse_dates=["game_date"], required_columns=required_columns)

    # Convert date to date object
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Validate data types and ranges
    for col in ["pts", "fg3m", "ast", "reb"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric, attempting conversion")
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Check for negative values
            if (df[col] < 0).any():
                n_negative = (df[col] < 0).sum()
                logger.warning(f"Found {n_negative} negative values in {col}, setting to 0")
                df.loc[df[col] < 0, col] = 0

    logger.info(f"Loaded game logs for {df['player_id'].nunique():,} players")

    return df


def load_prop_lines(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load player prop lines.

    Expected columns:
        - game_id: Unique game identifier
        - game_date: Date of the game
        - player_id: Unique player identifier
        - stat: Stat type ('pts', 'fg3m', 'ast', 'reb')
        - line: Prop line value (float)
        - over_odds: Decimal odds for over
        - under_odds: Decimal odds for under
        - book: Bookmaker name

    Args:
        path: Optional path to CSV file (defaults to RAW_DATA_DIR/player_prop_lines.csv)

    Returns:
        DataFrame with prop lines

    Raises:
        DataLoadError: If file doesn't exist or required columns are missing
    """
    if path is None:
        path = RAW_DATA_DIR / "player_prop_lines.csv"

    required_columns = ["game_id", "game_date", "player_id", "stat", "line"]

    df = _read_csv(path, parse_dates=["game_date"], required_columns=required_columns)

    # Convert date to date object
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Validate line values
    if (df["line"] < 0).any():
        n_negative = (df["line"] < 0).sum()
        logger.warning(f"Found {n_negative} negative line values, removing")
        df = df[df["line"] >= 0]

    # Validate odds if present
    for odds_col in ["over_odds", "under_odds"]:
        if odds_col in df.columns:
            if (df[odds_col] < 1.0).any():
                n_invalid = (df[odds_col] < 1.0).sum()
                logger.warning(f"Found {n_invalid} invalid odds in {odds_col} (< 1.0)")

    logger.info(f"Loaded {len(df):,} prop lines for {df['player_id'].nunique():,} players")

    return df


def load_bet_builder_legs(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load bet builder legs.

    Expected columns:
        - builder_id: Unique builder identifier
        - leg_id: Unique leg identifier within builder
        - game_id: Unique game identifier
        - game_date: Date of the game
        - player_id: Unique player identifier
        - stat: Stat type ('pts', 'fg3m', 'ast', 'reb')
        - direction: 'over' or 'under'
        - line: Line value used in the bet
        - builder_total_odds: Decimal odds for full builder
        - stake: Stake amount per builder

    Args:
        path: Optional path to CSV file (defaults to RAW_DATA_DIR/bet_builder_legs.csv)

    Returns:
        DataFrame with bet builder legs

    Raises:
        DataLoadError: If file doesn't exist or required columns are missing
    """
    if path is None:
        path = RAW_DATA_DIR / "bet_builder_legs.csv"

    required_columns = [
        "builder_id", "game_id", "game_date", "player_id",
        "stat", "direction", "line"
    ]

    df = _read_csv(path, parse_dates=["game_date"], required_columns=required_columns)

    # Convert date to date object
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Validate direction values
    valid_directions = {"over", "under"}
    df["direction"] = df["direction"].str.lower()
    invalid_directions = ~df["direction"].isin(valid_directions)
    if invalid_directions.any():
        n_invalid = invalid_directions.sum()
        logger.warning(f"Found {n_invalid} invalid direction values, removing")
        df = df[~invalid_directions]

    # Validate odds if present
    if "builder_total_odds" in df.columns:
        if (df["builder_total_odds"] < 1.0).any():
            n_invalid = (df["builder_total_odds"] < 1.0).sum()
            logger.warning(f"Found {n_invalid} invalid builder odds (< 1.0)")

    # Validate stake if present
    if "stake" in df.columns:
        if (df["stake"] <= 0).any():
            n_invalid = (df["stake"] <= 0).sum()
            logger.warning(f"Found {n_invalid} invalid stakes (<= 0), setting to 1.0")
            df.loc[df["stake"] <= 0, "stake"] = 1.0

    logger.info(
        f"Loaded {len(df):,} legs across {df['builder_id'].nunique():,} builders"
    )

    return df


def validate_data_consistency(
    player_games: pd.DataFrame,
    prop_lines: pd.DataFrame,
    bet_builder_legs: pd.DataFrame
) -> dict:
    """
    Validate consistency across datasets.

    Args:
        player_games: Player game logs
        prop_lines: Prop lines
        bet_builder_legs: Bet builder legs

    Returns:
        Dictionary with validation results and warnings
    """
    results = {
        "valid": True,
        "warnings": [],
        "stats": {}
    }

    # Check for missing games in game logs
    builder_games = set(bet_builder_legs["game_id"].unique())
    log_games = set(player_games["game_id"].unique())
    missing_games = builder_games - log_games

    if missing_games:
        results["warnings"].append(
            f"Found {len(missing_games)} games in builders but not in game logs"
        )

    # Check for missing players
    builder_players = set(bet_builder_legs["player_id"].unique())
    log_players = set(player_games["player_id"].unique())
    missing_players = builder_players - log_players

    if missing_players:
        results["warnings"].append(
            f"Found {len(missing_players)} players in builders but not in game logs"
        )

    # Summary stats
    results["stats"] = {
        "n_games_logs": player_games["game_id"].nunique(),
        "n_games_builders": bet_builder_legs["game_id"].nunique(),
        "n_players_logs": player_games["player_id"].nunique(),
        "n_players_builders": bet_builder_legs["player_id"].nunique(),
        "n_builders": bet_builder_legs["builder_id"].nunique(),
        "date_range_logs": (
            str(player_games["game_date"].min()),
            str(player_games["game_date"].max())
        ),
        "date_range_builders": (
            str(bet_builder_legs["game_date"].min()),
            str(bet_builder_legs["game_date"].max())
        ),
    }

    for warning in results["warnings"]:
        logger.warning(warning)

    if not results["warnings"]:
        logger.info("Data consistency validation passed")

    return results
