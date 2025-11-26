"""Feature engineering for bet builder analysis."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import get_logger, model_config


logger = get_logger(__name__)


def build_leg_level_dataset(
    player_games: pd.DataFrame,
    prop_lines: pd.DataFrame,
    bet_builder_legs: pd.DataFrame,
    stat_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Create a leg-level dataset that includes realised outcomes and hit flags.

    The function:
        - joins bet builder legs with player game logs
        - joins the official prop line used by the book
        - computes whether each leg hit its over/under
        - creates simple relative deviation features

    Args:
        player_games: DataFrame with player game logs
        prop_lines: DataFrame with prop lines
        bet_builder_legs: DataFrame with bet builder legs
        stat_cols: List of stat columns to process

    Returns:
        DataFrame with leg-level features and outcomes
    """
    stat_cols = stat_cols or model_config.stat_cols

    logger.info("Building leg-level dataset...")

    games = player_games.copy()
    legs = bet_builder_legs.copy()
    props = prop_lines.copy()

    # Map each leg to the realised stat from the game logs
    context_cols = [c for c in ["days_rest", "is_home"] if c in games.columns]
    base_cols = ["game_id", "game_date", "player_id"] + stat_cols + context_cols

    logger.info(f"Merging {len(legs):,} legs with game logs...")
    merged = legs.merge(
        games[base_cols],
        on=["game_id", "game_date", "player_id"],
        how="left",
        suffixes=("", "_realised"),
    )

    # Check for missing matches
    n_missing = merged[stat_cols[0]].isna().sum()
    if n_missing > 0:
        logger.warning(
            f"Found {n_missing} legs ({n_missing/len(merged)*100:.1f}%) "
            f"without matching game logs"
        )

    # Attach the bookmaker line that was actually available/used
    if not props.empty:
        logger.info(f"Merging with {len(props):,} prop lines...")
        merged = merged.merge(
            props[
                [
                    "game_id",
                    "game_date",
                    "player_id",
                    "stat",
                    "line",
                    "over_odds",
                    "under_odds",
                ]
            ],
            on=["game_id", "game_date", "player_id", "stat"],
            how="left",
            suffixes=("", "_book"),
        )
    else:
        merged["line_book"] = np.nan
        merged["over_odds"] = np.nan
        merged["under_odds"] = np.nan

    # Realised stat value for the leg
    merged["realised"] = np.nan
    for s in stat_cols:
        mask = merged["stat"] == s
        merged.loc[mask, "realised"] = merged.loc[mask, s]

    # Use leg-specific line if present, otherwise bookmaker line
    if "line_book" in merged.columns:
        merged["effective_line"] = merged["line"].fillna(merged["line_book"])
    else:
        merged["effective_line"] = merged["line"]

    # Determine hit for each leg
    logger.info("Computing leg outcomes...")
    merged["direction_lower"] = merged["direction"].str.lower()
    merged["leg_hit"] = np.where(
        merged["direction_lower"] == "over",
        (merged["realised"] > merged["effective_line"]).astype(int),
        (merged["realised"] < merged["effective_line"]).astype(int),
    )

    # Simple deviation feature: realised minus line
    merged["realised_minus_line"] = merged["realised"] - merged["effective_line"]

    # Add percentage deviation
    merged["realised_pct_diff"] = (
        merged["realised_minus_line"] / merged["effective_line"].replace(0, np.nan)
    ) * 100

    # Log summary stats
    logger.info(f"Leg-level dataset created: {len(merged):,} legs")
    logger.info(f"Overall leg hit rate: {merged['leg_hit'].mean()*100:.2f}%")

    direction_hit_rates = merged.groupby("direction_lower")["leg_hit"].mean() * 100
    for direction, rate in direction_hit_rates.items():
        logger.info(f"  {direction.capitalize()} hit rate: {rate:.2f}%")

    return merged


def build_builder_level_dataset(leg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate legs to builder-level outcomes and features.

    For each builder_id we compute:
        - whether all legs hit
        - number of legs
        - mean / max absolute edge (realised_minus_line)
        - carry forward stake and total odds
        - bookmaker implied probability from decimal odds

    Args:
        leg_df: DataFrame with leg-level data

    Returns:
        DataFrame with builder-level features and outcomes
    """
    logger.info("Building builder-level dataset...")

    df = leg_df.copy()
    df["abs_realised_minus_line"] = df["realised_minus_line"].abs()

    logger.info(f"Aggregating {df['builder_id'].nunique():,} unique builders...")

    agg = df.groupby("builder_id").agg(
        game_date=("game_date", "first"),
        game_id=("game_id", "first"),
        n_legs=("leg_id", "nunique"),
        builder_hit=("leg_hit", "min"),  # =1 only if all legs are 1
        mean_abs_edge=("abs_realised_minus_line", "mean"),
        max_abs_edge=("abs_realised_minus_line", "max"),
        stake=("stake", "first"),
        decimal_odds=("builder_total_odds", "first"),
        n_over=("direction_lower", lambda x: (x == "over").sum()),
        n_under=("direction_lower", lambda x: (x == "under").sum()),
    )

    agg.reset_index(inplace=True)

    # If stake is missing, assume unit stake
    agg["stake"] = agg["stake"].fillna(1.0)

    # Bookmaker implied probability (ignoring overround)
    agg["book_implied_prob"] = 1.0 / agg["decimal_odds"]

    # Add leg composition features
    agg["pct_over"] = agg["n_over"] / agg["n_legs"]
    agg["pct_under"] = agg["n_under"] / agg["n_legs"]
    agg["is_all_over"] = (agg["n_over"] == agg["n_legs"]).astype(int)
    agg["is_all_under"] = (agg["n_under"] == agg["n_legs"]).astype(int)
    agg["is_mixed"] = ((agg["n_over"] > 0) & (agg["n_under"] > 0)).astype(int)

    # Log summary stats
    logger.info(f"Builder-level dataset created: {len(agg):,} builders")
    logger.info(f"Overall builder hit rate: {agg['builder_hit'].mean()*100:.2f}%")
    logger.info(f"Average legs per builder: {agg['n_legs'].mean():.2f}")
    logger.info(f"Leg distribution: {agg['n_legs'].value_counts().sort_index().to_dict()}")

    return agg


def add_stat_composition_features(
    leg_df: pd.DataFrame,
    builder_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add features about which stats are included in each builder.

    Args:
        leg_df: DataFrame with leg-level data
        builder_df: DataFrame with builder-level data

    Returns:
        Builder DataFrame with added stat composition features
    """
    logger.info("Adding stat composition features...")

    # Count stats per builder
    stat_counts = leg_df.groupby(["builder_id", "stat"]).size().unstack(fill_value=0)

    # Create binary indicators for each stat
    for stat in model_config.stat_cols:
        if stat in stat_counts.columns:
            builder_df[f"has_{stat}"] = (
                builder_df["builder_id"].map(stat_counts[stat]) > 0
            ).astype(int)
        else:
            builder_df[f"has_{stat}"] = 0

    # Count unique stats
    builder_df["n_unique_stats"] = (
        builder_df[[f"has_{stat}" for stat in model_config.stat_cols]].sum(axis=1)
    )

    logger.info("Stat composition features added")

    return builder_df


def add_player_diversity_features(
    leg_df: pd.DataFrame,
    builder_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add features about player diversity within builders.

    Args:
        leg_df: DataFrame with leg-level data
        builder_df: DataFrame with builder-level data

    Returns:
        Builder DataFrame with added player diversity features
    """
    logger.info("Adding player diversity features...")

    player_counts = leg_df.groupby("builder_id")["player_id"].agg(
        n_unique_players="nunique",
        n_total_legs="count"
    )

    builder_df = builder_df.merge(player_counts, on="builder_id", how="left")

    # Fraction of legs that are same-player
    builder_df["same_player_ratio"] = (
        builder_df["n_total_legs"] / builder_df["n_unique_players"]
    )

    # Binary indicator for single-player builders
    builder_df["is_single_player"] = (
        builder_df["n_unique_players"] == 1
    ).astype(int)

    logger.info("Player diversity features added")

    return builder_df


def add_correlation_risk_features(
    leg_df: pd.DataFrame,
    builder_df: pd.DataFrame,
    correlation_matrix: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Add features related to correlation risk between legs.

    Args:
        leg_df: DataFrame with leg-level data
        builder_df: DataFrame with builder-level data
        correlation_matrix: Optional correlation matrix between stats

    Returns:
        Builder DataFrame with added correlation risk features
    """
    if correlation_matrix is None:
        logger.info("No correlation matrix provided, skipping correlation features")
        return builder_df

    logger.info("Adding correlation risk features...")

    # This is a simplified version - in production you'd want to compute
    # average pairwise correlations for each builder's specific legs
    builder_df["high_correlation_risk"] = builder_df["is_single_player"].copy()

    logger.info("Correlation risk features added")

    return builder_df
