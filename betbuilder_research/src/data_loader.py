"""Data loading module with validation and error handling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

from .config import RAW_DATA_DIR, get_logger


logger = get_logger(__name__)

# Column mapping from PlayerStatistics format to pipeline format
COLUMN_MAPPING = {
    'personId': 'player_id',
    'gameId': 'game_id',
    'gameDateTimeEst': 'game_date',
    'points': 'pts',
    'threePointersMade': 'fg3m',
    'assists': 'ast',
    'reboundsTotal': 'reb',
    'firstName': 'first_name',
    'lastName': 'last_name',
    'playerteamName': 'team',
    'opponentteamName': 'opponent',
    'home': 'home_away',
    'numMinutes': 'minutes',
    'steals': 'stl',
    'blocks': 'blk',
    'turnovers': 'tov',
    'fieldGoalsMade': 'fgm',
    'fieldGoalsAttempted': 'fga',
    'freeThrowsMade': 'ftm',
    'freeThrowsAttempted': 'fta',
}


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

    Automatically detects and loads from PlayerStatistics_*.csv files if
    player_game_logs.csv doesn't exist, applying column mapping.

    Expected output columns:
        - game_id: Unique game identifier
        - game_date: Date of the game
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
        path: Optional path to CSV file

    Returns:
        DataFrame with player game logs

    Raises:
        DataLoadError: If no suitable file exists
    """
    if path is None:
        path = RAW_DATA_DIR / "player_game_logs.csv"

    if not path.exists() and path != RAW_DATA_DIR / "player_game_logs.csv":
        raise DataLoadError(f"Explicit input file not found: {path}")

    # Try loading the standard file first
    if path.exists():
        required_columns = [
            "game_id", "game_date", "player_id", "pts", "fg3m", "ast", "reb"
        ]
        df = _read_csv(path, parse_dates=["game_date"], required_columns=required_columns)
    else:
        # Look for PlayerStatistics files
        df = _load_player_statistics_file()

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


def _load_player_statistics_file() -> pd.DataFrame:
    """
    Load from PlayerStatistics_*.csv files and apply column mapping.

    Returns:
        DataFrame with standardized column names
    """
    # Try different file names in order of preference
    candidates = [
        RAW_DATA_DIR / "PlayerStatistics_2025-2026_Feb.csv",
        RAW_DATA_DIR / "PlayerStatistics_2018_2025.csv",
        RAW_DATA_DIR / "PlayerStatistics_2024_2025.csv",
        RAW_DATA_DIR / "PlayerStatistics.csv",
    ]

    source_path = None
    for candidate in candidates:
        if candidate.exists():
            source_path = candidate
            break

    if source_path is None:
        raise DataLoadError(
            f"No player statistics file found. Looked for:\n"
            + "\n".join(f"  - {c}" for c in candidates)
        )

    logger.info(f"Loading from {source_path.name} (auto-detected)")

    # Load with date parsing
    df = pd.read_csv(source_path, parse_dates=["gameDateTimeEst"])
    logger.info(f"Loaded {len(df):,} rows from {source_path.name}")

    # Apply column mapping
    df = df.rename(columns=COLUMN_MAPPING)

    # Create player_name if not present
    if "player_name" not in df.columns and "first_name" in df.columns:
        df["player_name"] = df["first_name"] + " " + df["last_name"]

    # Ensure required columns exist
    required = ["game_id", "game_date", "player_id", "pts", "fg3m", "ast", "reb"]
    missing = set(required) - set(df.columns)
    if missing:
        raise DataLoadError(f"Missing required columns after mapping: {missing}")

    return df


def load_prop_lines(
    path: Optional[Path] = None,
    player_games: Optional[pd.DataFrame] = None,
    learned_model_path: Optional[Path] = None,
    *, allow_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Load player prop lines.

    Missing files fail by default. Synthetic generation requires allow_synthetic=True.
    If a learned model path is provided, uses the trained model to generate lines.

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
        player_games: Optional player games DataFrame for synthetic generation
        learned_model_path: Optional path to trained LearnedLineGenerator model

    Returns:
        DataFrame with prop lines
    """
    if path is None:
        path = RAW_DATA_DIR / "player_prop_lines.csv"

    if path.exists():
        required_columns = ["game_id", "game_date", "player_id", "stat", "line"]
        df = _read_csv(path, parse_dates=["game_date"], required_columns=required_columns)
    elif learned_model_path is not None and Path(learned_model_path).exists():
        # Use learned model for line generation
        logger.info(f"Using learned model from {learned_model_path}")
        from .models.line_generator import LearnedLineGenerator
        if player_games is None:
            player_games = load_player_game_logs()
        model = LearnedLineGenerator.load(learned_model_path)
        df = model.generate_lines(player_games)
    else:
        if not allow_synthetic:
            raise DataLoadError(f"Real prop lines required: {path}; synthetic generation is disabled")
        logger.warning("EXPLICIT SYNTHETIC DEMO: generating prop lines")
        if player_games is None:
            player_games = load_player_game_logs()
        df = _generate_synthetic_prop_lines(player_games)

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


def _generate_synthetic_prop_lines(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic prop lines based on player historical averages.

    Lines are set around player averages with some noise to simulate
    realistic betting lines.

    Args:
        player_games: DataFrame with player game logs

    Returns:
        DataFrame with synthetic prop lines
    """
    logger.info("Generating synthetic prop lines from player averages...")

    stats = ["pts", "fg3m", "ast", "reb"]
    records = []

    # Calculate rolling averages for each player
    player_games = player_games.sort_values(["player_id", "game_date"])

    for player_id in player_games["player_id"].unique():
        player_df = player_games[player_games["player_id"] == player_id].copy()

        if len(player_df) < 5:
            continue

        # Calculate rolling mean (last 10 games)
        for stat in stats:
            if stat not in player_df.columns:
                continue

            player_df[f"{stat}_avg"] = (
                player_df[stat]
                .rolling(window=10, min_periods=3)
                .mean()
                .shift(1)  # Use previous games only
            )

        # Generate prop lines for each game after initial period
        for idx, row in player_df.iloc[5:].iterrows():
            for stat in stats:
                avg_col = f"{stat}_avg"
                if avg_col not in player_df.columns or pd.isna(row[avg_col]):
                    continue

                avg = row[avg_col]
                if avg <= 0:
                    continue

                # Add some noise to simulate bookmaker line setting
                # Lines typically round to 0.5
                noise = np.random.normal(0, avg * 0.1)
                line = round((avg + noise) * 2) / 2  # Round to nearest 0.5

                # Generate odds (slightly favoring under to simulate vig)
                over_odds = round(np.random.uniform(1.85, 1.95), 2)
                under_odds = round(np.random.uniform(1.85, 1.95), 2)

                records.append({
                    "game_id": row["game_id"],
                    "game_date": row["game_date"],
                    "player_id": player_id,
                    "stat": stat,
                    "line": max(0.5, line),  # Ensure positive line
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "book": "synthetic",
                })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df):,} synthetic prop lines")

    return df


def load_bet_builder_legs(
    path: Optional[Path] = None,
    prop_lines: Optional[pd.DataFrame] = None,
    player_games: Optional[pd.DataFrame] = None,
    *, allow_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Load bet builder legs.

    Missing files fail by default. Synthetic generation requires allow_synthetic=True.

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
        prop_lines: Optional prop lines DataFrame for synthetic generation
        player_games: Optional player games DataFrame for synthetic generation

    Returns:
        DataFrame with bet builder legs
    """
    if path is None:
        path = RAW_DATA_DIR / "bet_builder_legs.csv"

    if path.exists():
        required_columns = [
            "builder_id", "game_id", "game_date", "player_id",
            "stat", "direction", "line"
        ]
        df = _read_csv(path, parse_dates=["game_date"], required_columns=required_columns)
    else:
        if not allow_synthetic:
            raise DataLoadError(f"Real builder legs required: {path}; synthetic generation is disabled")
        logger.warning("EXPLICIT SYNTHETIC DEMO: generating builder legs")
        if player_games is None:
            player_games = load_player_game_logs()
        if prop_lines is None:
            prop_lines = load_prop_lines(player_games=player_games, allow_synthetic=True)
        df = _generate_synthetic_bet_builders(prop_lines, player_games)

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


def _generate_synthetic_bet_builders(
    prop_lines: pd.DataFrame,
    player_games: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate synthetic bet builder legs by combining prop lines.

    Creates multi-leg builders (2-4 legs) from available prop lines,
    simulating realistic bet builder construction.

    Args:
        prop_lines: DataFrame with prop lines
        player_games: DataFrame with player game logs

    Returns:
        DataFrame with synthetic bet builder legs
    """
    logger.info("Generating synthetic bet builders from prop lines...")

    np.random.seed(42)  # For reproducibility

    # Get unique game dates with enough prop lines
    games_with_props = (
        prop_lines.groupby(["game_id", "game_date"])
        .size()
        .reset_index(name="n_props")
    )
    games_with_props = games_with_props[games_with_props["n_props"] >= 4]

    records = []
    builder_id = 0

    # Sample games for builders (limit to avoid too much data)
    sample_size = min(len(games_with_props), 5000)
    sampled_games = games_with_props.sample(n=sample_size, random_state=42)

    for _, game_row in sampled_games.iterrows():
        game_id = game_row["game_id"]
        game_date = game_row["game_date"]

        # Get props for this game
        game_props = prop_lines[prop_lines["game_id"] == game_id]

        if len(game_props) < 2:
            continue

        # Create 1-3 builders per game
        n_builders = np.random.randint(1, 4)

        for _ in range(n_builders):
            # Random number of legs (2-4)
            n_legs = np.random.randint(2, min(5, len(game_props) + 1))

            # Sample props for this builder
            builder_props = game_props.sample(n=n_legs, replace=False)

            # Calculate combined odds (multiply individual leg odds)
            leg_odds = []
            for leg_idx, (_, prop) in enumerate(builder_props.iterrows()):
                # Random direction with slight over bias
                direction = "over" if np.random.random() < 0.55 else "under"
                leg_odd = prop["over_odds"] if direction == "over" else prop["under_odds"]
                leg_odds.append(leg_odd)

                records.append({
                    "builder_id": builder_id,
                    "leg_id": leg_idx,
                    "game_id": game_id,
                    "game_date": game_date,
                    "player_id": prop["player_id"],
                    "stat": prop["stat"],
                    "direction": direction,
                    "line": prop["line"],
                    "leg_odds": leg_odd,
                })

            # Update builder_total_odds for all legs in this builder
            total_odds = np.prod(leg_odds)
            for i in range(n_legs):
                records[-(n_legs - i)]["builder_total_odds"] = round(total_odds, 2)
                records[-(n_legs - i)]["stake"] = 1.0

            builder_id += 1

    df = pd.DataFrame(records)

    # Add actual outcomes based on player game data
    if len(df) > 0:
        df = _add_leg_outcomes(df, player_games)

    logger.info(f"Generated {len(df):,} legs across {builder_id:,} builders")

    return df


def _add_leg_outcomes(legs_df: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add actual outcomes to bet builder legs based on player game data.

    Args:
        legs_df: DataFrame with bet builder legs
        player_games: DataFrame with player game logs

    Returns:
        DataFrame with leg_hit column added
    """
    # Create lookup for actual stats
    stat_lookup = player_games.set_index(["game_id", "player_id"])[
        ["pts", "fg3m", "ast", "reb"]
    ].to_dict("index")

    def check_hit(row):
        key = (row["game_id"], row["player_id"])
        if key not in stat_lookup:
            return np.nan

        actual = stat_lookup[key].get(row["stat"], np.nan)
        if pd.isna(actual):
            return np.nan

        if row["direction"] == "over":
            return 1 if actual > row["line"] else 0
        else:
            return 1 if actual < row["line"] else 0

    legs_df["leg_hit"] = legs_df.apply(check_hit, axis=1)

    # Remove legs without outcomes
    n_before = len(legs_df)
    legs_df = legs_df.dropna(subset=["leg_hit"])
    legs_df["leg_hit"] = legs_df["leg_hit"].astype(int)

    n_removed = n_before - len(legs_df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed:,} legs without matching game data")

    return legs_df


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
