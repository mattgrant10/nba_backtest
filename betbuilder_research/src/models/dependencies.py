"""Analyze dependencies and correlations between player stats."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import get_logger, model_config


logger = get_logger(__name__)


def compute_player_stat_correlations(
    player_games: pd.DataFrame,
    stat_cols: Iterable[str] | None = None,
    min_games: int | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute per-player correlation matrices between stats, flattened to long format.

    Returns columns:
        - player_id
        - stat_x
        - stat_y
        - corr
        - n_games
        - method

    Args:
        player_games: DataFrame with player game logs
        stat_cols: Stats to analyze
        min_games: Minimum games required
        method: Correlation method ('spearman', 'pearson')

    Returns:
        DataFrame with per-player correlations
    """
    stat_cols = stat_cols or model_config.stat_cols
    min_games = min_games or model_config.min_games

    logger.info(
        f"Computing per-player correlations using {method} method, "
        f"min_games={min_games}"
    )

    records: List[dict] = []

    for player_id, grp in tqdm(
        player_games.groupby("player_id"),
        desc="Players",
        leave=True
    ):
        grp_stats = grp[list(stat_cols)].dropna()
        n = len(grp_stats)
        if n < min_games:
            continue

        try:
            corr_matrix = grp_stats.corr(method=method)

            for i, stat_x in enumerate(stat_cols):
                for j, stat_y in enumerate(stat_cols):
                    if j < i:  # upper triangle only
                        continue

                    corr = corr_matrix.loc[stat_x, stat_y]

                    records.append(
                        {
                            "player_id": player_id,
                            "stat_x": stat_x,
                            "stat_y": stat_y,
                            "corr": float(corr),
                            "n_games": n,
                            "method": method,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error computing correlation for player {player_id}: {e}")
            continue

    df = pd.DataFrame.from_records(records)
    logger.info(
        f"Computed correlations for {df['player_id'].nunique():,} players, "
        f"{len(df):,} total correlations"
    )

    return df


def compute_global_correlation(
    player_games: pd.DataFrame,
    stat_cols: Iterable[str] | None = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute a global correlation matrix across all players and games.

    Args:
        player_games: DataFrame with player game logs
        stat_cols: Stats to analyze
        method: Correlation method ('spearman', 'pearson')

    Returns:
        DataFrame with global correlations in long format
    """
    stat_cols = stat_cols or model_config.stat_cols

    logger.info(f"Computing global correlations using {method} method")

    df = player_games[list(stat_cols)].dropna()

    logger.info(f"Using {len(df):,} observations for global correlation")

    try:
        corr_matrix = df.corr(method=method)
    except Exception as e:
        logger.error(f"Error computing global correlation: {e}")
        return pd.DataFrame()

    records: List[dict] = []
    for i, stat_x in enumerate(stat_cols):
        for j, stat_y in enumerate(stat_cols):
            if j < i:  # upper triangle only
                continue

            corr = corr_matrix.loc[stat_x, stat_y]

            records.append(
                {
                    "stat_x": stat_x,
                    "stat_y": stat_y,
                    "corr": float(corr),
                    "n_obs": len(df),
                    "method": method,
                }
            )

    result_df = pd.DataFrame.from_records(records)
    logger.info(f"Global correlation matrix computed with {len(result_df)} pairs")

    # Log strongest correlations
    non_diag = result_df[result_df["stat_x"] != result_df["stat_y"]].copy()
    if len(non_diag) > 0:
        strongest = non_diag.nlargest(3, "corr")
        logger.info("Strongest global correlations:")
        for _, row in strongest.iterrows():
            logger.info(
                f"  {row['stat_x']} <-> {row['stat_y']}: {row['corr']:.3f}"
            )

    return result_df


def analyze_correlation_patterns(
    player_corr_df: pd.DataFrame,
    global_corr_df: pd.DataFrame
) -> dict:
    """
    Analyze patterns in correlation data.

    Args:
        player_corr_df: Per-player correlations
        global_corr_df: Global correlations

    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing correlation patterns...")

    results = {}

    # Filter out diagonal (self-correlations)
    player_off_diag = player_corr_df[
        player_corr_df["stat_x"] != player_corr_df["stat_y"]
    ].copy()

    global_off_diag = global_corr_df[
        global_corr_df["stat_x"] != global_corr_df["stat_y"]
    ].copy()

    # Summary statistics for each stat pair
    if len(player_off_diag) > 0:
        summary = player_off_diag.groupby(["stat_x", "stat_y"])["corr"].agg([
            "mean", "median", "std", "min", "max"
        ]).reset_index()

        summary.columns = ["stat_x", "stat_y", "mean_corr", "median_corr",
                          "std_corr", "min_corr", "max_corr"]

        results["per_player_summary"] = summary

        logger.info("Average correlations across players:")
        for _, row in summary.iterrows():
            logger.info(
                f"  {row['stat_x']} <-> {row['stat_y']}: "
                f"mean={row['mean_corr']:.3f}, std={row['std_corr']:.3f}"
            )

    # Compare player-level to global
    if len(global_off_diag) > 0 and len(player_off_diag) > 0:
        comparison = []
        for _, g_row in global_off_diag.iterrows():
            stat_x = g_row["stat_x"]
            stat_y = g_row["stat_y"]

            player_subset = player_off_diag[
                ((player_off_diag["stat_x"] == stat_x) &
                 (player_off_diag["stat_y"] == stat_y)) |
                ((player_off_diag["stat_x"] == stat_y) &
                 (player_off_diag["stat_y"] == stat_x))
            ]

            if len(player_subset) > 0:
                comparison.append({
                    "stat_x": stat_x,
                    "stat_y": stat_y,
                    "global_corr": g_row["corr"],
                    "player_mean_corr": player_subset["corr"].mean(),
                    "player_median_corr": player_subset["corr"].median(),
                    "player_std_corr": player_subset["corr"].std(),
                })

        results["comparison"] = pd.DataFrame(comparison)

        logger.info("Global vs player-level correlations:")
        for item in comparison:
            logger.info(
                f"  {item['stat_x']} <-> {item['stat_y']}: "
                f"global={item['global_corr']:.3f}, "
                f"player_mean={item['player_mean_corr']:.3f}"
            )

    return results


def identify_high_correlation_pairs(
    player_corr_df: pd.DataFrame,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Identify player-stat pairs with high correlations.

    Args:
        player_corr_df: Per-player correlations
        threshold: Minimum absolute correlation to flag

    Returns:
        DataFrame with high correlation pairs
    """
    logger.info(f"Identifying correlations >= {threshold}")

    # Filter out diagonal
    off_diag = player_corr_df[
        player_corr_df["stat_x"] != player_corr_df["stat_y"]
    ].copy()

    # Filter by threshold
    high_corr = off_diag[off_diag["corr"].abs() >= threshold].copy()

    logger.info(
        f"Found {len(high_corr):,} player-stat pairs with |corr| >= {threshold}"
    )

    if len(high_corr) > 0:
        # Count by stat pair
        pair_counts = high_corr.groupby(["stat_x", "stat_y"]).size().reset_index()
        pair_counts.columns = ["stat_x", "stat_y", "n_players"]
        pair_counts = pair_counts.sort_values("n_players", ascending=False)

        logger.info("Most common high-correlation pairs:")
        for _, row in pair_counts.head(5).iterrows():
            logger.info(
                f"  {row['stat_x']} <-> {row['stat_y']}: "
                f"{row['n_players']} players"
            )

    return high_corr
