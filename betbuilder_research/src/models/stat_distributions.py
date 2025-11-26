"""Statistical distribution modeling for player stats."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from tqdm import tqdm

from ..config import get_logger, model_config


logger = get_logger(__name__)


def summarise_player_stat_distributions(
    player_games: pd.DataFrame,
    stat_cols: Iterable[str] | None = None,
    min_games: int | None = None,
) -> pd.DataFrame:
    """
    Compute empirical distribution summaries for each player and stat.

    Returns a DataFrame with:
        - player_id
        - stat
        - n_games
        - mean
        - std
        - var
        - p10, p25, p50, p75, p90

    Args:
        player_games: DataFrame with player game logs
        stat_cols: Stats to analyze
        min_games: Minimum games required per player

    Returns:
        DataFrame with distribution summaries
    """
    stat_cols = stat_cols or model_config.stat_cols
    min_games = min_games or model_config.min_games

    logger.info(
        f"Computing distributions for {len(stat_cols)} stats, "
        f"min_games={min_games}"
    )

    records: List[dict] = []

    for stat in stat_cols:
        logger.info(f"Processing stat: {stat}")
        grouped = player_games.groupby("player_id")[stat]

        for player_id, series in tqdm(grouped, desc=f"Players ({stat})", leave=False):
            series = series.dropna()
            n = len(series)
            if n < min_games:
                continue

            values = series.to_numpy()

            rec = {
                "player_id": player_id,
                "stat": stat,
                "n_games": n,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "var": float(np.var(values, ddof=1)),
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    logger.info(
        f"Created distribution summaries for {len(df):,} player-stat combinations"
    )

    return df


def fit_parametric_count_model(mean: float, var: float) -> dict:
    """
    Fit a simple parametric distribution (Poisson or Negative Binomial).

    If var H mean we use Poisson.
    If var > mean we use a Negative Binomial with method-of-moments parameters.

    Args:
        mean: Sample mean
        var: Sample variance

    Returns:
        Dictionary with family and parameters
    """
    if mean <= 0:
        return {"family": "degenerate", "params": {"loc": 0}}

    # Poisson if variance is close to mean (allowing 10% tolerance)
    if np.isclose(var, mean, rtol=0.1) or var < mean:
        return {"family": "poisson", "params": {"mu": float(mean)}}

    # Negative binomial parameterisation:
    # mean = r * (1-p) / p
    # var = r * (1-p) / p**2
    # Solving: r = mean**2 / (var - mean)
    try:
        r = mean**2 / (var - mean)
        p = r / (r + mean)

        # Validate parameters
        if r <= 0 or p <= 0 or p >= 1:
            logger.warning(
                f"Invalid NB parameters (r={r:.3f}, p={p:.3f}), "
                f"falling back to Poisson"
            )
            return {"family": "poisson", "params": {"mu": float(mean)}}

        return {"family": "nbinom", "params": {"n": float(r), "p": float(p)}}

    except (ZeroDivisionError, ValueError) as e:
        logger.warning(f"Error fitting NB model: {e}, falling back to Poisson")
        return {"family": "poisson", "params": {"mu": float(mean)}}


def attach_parametric_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the empirical summary, attach a simple parametric model choice.

    Args:
        summary_df: DataFrame with distribution summaries

    Returns:
        DataFrame with added family and params columns
    """
    logger.info("Fitting parametric models...")

    summary_df = summary_df.copy()
    families: List[str] = []
    params_list: List[dict] = []

    for _, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Fitting models"):
        model = fit_parametric_count_model(row["mean"], row["var"])
        families.append(model["family"])
        params_list.append(model["params"])

    summary_df["family"] = families
    summary_df["params"] = params_list

    # Log distribution of model families
    family_counts = summary_df["family"].value_counts()
    logger.info("Model family distribution:")
    for family, count in family_counts.items():
        pct = count / len(summary_df) * 100
        logger.info(f"  {family}: {count} ({pct:.1f}%)")

    return summary_df


def _tail_probs_for_line(
    family: str,
    params: dict,
    line: float
) -> tuple[float, float]:
    """
    Helper: compute P(X > line), P(X < line) for a discrete count X.

    For counts with half-point lines:
        over line L:  X >= ceil(L + epsilon)
        under line L: X <= floor(L - epsilon)

    Args:
        family: Distribution family
        params: Distribution parameters
        line: Line value

    Returns:
        Tuple of (prob_over, prob_under)
    """
    k_under = int(np.floor(line - 1e-9))
    k_over_start = k_under + 1

    try:
        if family == "poisson":
            mu = params["mu"]
            cdf_under = poisson.cdf(k_under, mu)
            cdf_before_over = poisson.cdf(k_over_start - 1, mu)
            prob_under = float(cdf_under)
            prob_over = float(1.0 - cdf_before_over)

        elif family == "nbinom":
            n = params["n"]
            p = params["p"]
            cdf_under = nbinom.cdf(k_under, n, p)
            cdf_before_over = nbinom.cdf(k_over_start - 1, n, p)
            prob_under = float(cdf_under)
            prob_over = float(1.0 - cdf_before_over)

        else:
            prob_under = np.nan
            prob_over = np.nan

    except Exception as e:
        logger.warning(f"Error computing tail probabilities: {e}")
        prob_under = np.nan
        prob_over = np.nan

    return prob_over, prob_under


def compute_tail_probabilities(
    summary_with_models: pd.DataFrame,
    prop_lines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate P(X > line) for overs and P(X < line) for unders using parametric models.

    Returns a DataFrame with:
        - player_id
        - stat
        - game_id
        - game_date
        - line
        - prob_over
        - prob_under
        - implied_odds_over
        - implied_odds_under

    Args:
        summary_with_models: DataFrame with fitted models
        prop_lines: DataFrame with prop lines

    Returns:
        DataFrame with tail probabilities
    """
    logger.info("Computing tail probabilities...")

    models = summary_with_models[["player_id", "stat", "family", "params"]].copy()

    df = prop_lines.merge(models, on=["player_id", "stat"], how="left")

    # Track how many lines have models
    n_with_models = df["family"].notna().sum()
    logger.info(
        f"Found models for {n_with_models:,} / {len(df):,} "
        f"({n_with_models/len(df)*100:.1f}%) prop lines"
    )

    records: List[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing probabilities"):
        family = row["family"]
        params = row["params"]
        line = row["line"]

        if pd.isna(family) or params is None:
            continue

        prob_over, prob_under = _tail_probs_for_line(family, params, line)

        # Convert to implied odds
        implied_odds_over = 1 / prob_over if prob_over > 0 else np.nan
        implied_odds_under = 1 / prob_under if prob_under > 0 else np.nan

        records.append(
            {
                "player_id": row["player_id"],
                "stat": row["stat"],
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "line": line,
                "prob_over": float(prob_over),
                "prob_under": float(prob_under),
                "implied_odds_over": float(implied_odds_over),
                "implied_odds_under": float(implied_odds_under),
            }
        )

    result_df = pd.DataFrame.from_records(records)
    logger.info(f"Computed probabilities for {len(result_df):,} lines")

    return result_df


def compute_value_bets(
    tail_probs: pd.DataFrame,
    prop_lines: pd.DataFrame,
    min_edge: float = 0.05
) -> pd.DataFrame:
    """
    Identify value bets where model probabilities suggest positive expected value.

    Args:
        tail_probs: DataFrame with model-implied probabilities
        prop_lines: DataFrame with bookmaker odds
        min_edge: Minimum edge required (default 5%)

    Returns:
        DataFrame with value bets
    """
    logger.info(f"Identifying value bets with min edge={min_edge*100:.1f}%...")

    # Merge probabilities with bookmaker odds
    df = tail_probs.merge(
        prop_lines[["game_id", "player_id", "stat", "over_odds", "under_odds"]],
        on=["game_id", "player_id", "stat"],
        how="inner"
    )

    # Calculate edges
    df["edge_over"] = (df["prob_over"] * df["over_odds"]) - 1
    df["edge_under"] = (df["prob_under"] * df["under_odds"]) - 1

    # Filter for value bets
    value_bets = df[
        (df["edge_over"] >= min_edge) | (df["edge_under"] >= min_edge)
    ].copy()

    # Determine best direction
    value_bets["best_direction"] = np.where(
        value_bets["edge_over"] > value_bets["edge_under"],
        "over",
        "under"
    )
    value_bets["best_edge"] = np.maximum(
        value_bets["edge_over"],
        value_bets["edge_under"]
    )

    logger.info(f"Found {len(value_bets):,} value bets")

    if len(value_bets) > 0:
        logger.info(f"Average edge: {value_bets['best_edge'].mean()*100:.2f}%")
        logger.info(f"Median edge: {value_bets['best_edge'].median()*100:.2f}%")

    return value_bets
