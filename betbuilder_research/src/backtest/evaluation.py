"""Evaluation metrics for bet builder backtesting."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ..config import get_logger


logger = get_logger(__name__)


def compute_roi(
    hits: np.ndarray,
    decimal_odds: np.ndarray,
    stakes: np.ndarray | float = 1.0,
) -> float:
    """
    Calculate return on investment.

    ROI = (total profit) / (total stake)

    Args:
        hits: Binary array of hits (1) or misses (0)
        decimal_odds: Array of decimal odds
        stakes: Array of stakes or single stake value

    Returns:
        ROI as a decimal (e.g., 0.10 = 10% ROI)
    """
    stakes_arr = np.asarray(stakes, dtype=float)
    if stakes_arr.shape == ():
        stakes_arr = np.full_like(decimal_odds, fill_value=stakes_arr, dtype=float)

    # Calculate profit/loss for each bet
    profits = np.where(
        hits == 1,
        stakes_arr * (decimal_odds - 1.0),  # Win: stake * (odds - 1)
        -stakes_arr  # Loss: -stake
    )

    total_profit = profits.sum()
    total_stake = stakes_arr.sum()

    if total_stake == 0:
        return 0.0

    roi = total_profit / total_stake
    return float(roi)


def compute_profit(
    hits: np.ndarray,
    decimal_odds: np.ndarray,
    stakes: np.ndarray | float = 1.0,
) -> float:
    """
    Calculate total profit/loss.

    Args:
        hits: Binary array of hits (1) or misses (0)
        decimal_odds: Array of decimal odds
        stakes: Array of stakes or single stake value

    Returns:
        Total profit (can be negative)
    """
    stakes_arr = np.asarray(stakes, dtype=float)
    if stakes_arr.shape == ():
        stakes_arr = np.full_like(decimal_odds, fill_value=stakes_arr, dtype=float)

    profits = np.where(
        hits == 1,
        stakes_arr * (decimal_odds - 1.0),
        -stakes_arr
    )

    return float(profits.sum())


def compute_yield(
    hits: np.ndarray,
    decimal_odds: np.ndarray,
    stakes: np.ndarray | float = 1.0,
) -> float:
    """
    Calculate yield (profit per bet).

    Yield = total profit / number of bets

    Args:
        hits: Binary array of hits (1) or misses (0)
        decimal_odds: Array of decimal odds
        stakes: Array of stakes or single stake value

    Returns:
        Average profit per bet
    """
    profit = compute_profit(hits, decimal_odds, stakes)
    n_bets = len(hits)

    if n_bets == 0:
        return 0.0

    return float(profit / n_bets)


def compute_sharpe_ratio(
    hits: np.ndarray,
    decimal_odds: np.ndarray,
    stakes: np.ndarray | float = 1.0,
) -> float:
    """
    Calculate Sharpe ratio for betting returns.

    Sharpe = mean(returns) / std(returns)

    Args:
        hits: Binary array of hits (1) or misses (0)
        decimal_odds: Array of decimal odds
        stakes: Array of stakes or single stake value

    Returns:
        Sharpe ratio
    """
    stakes_arr = np.asarray(stakes, dtype=float)
    if stakes_arr.shape == ():
        stakes_arr = np.full_like(decimal_odds, fill_value=stakes_arr, dtype=float)

    # Calculate returns (profit/loss per stake)
    returns = np.where(
        hits == 1,
        decimal_odds - 1.0,
        -1.0
    )

    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    return float(mean_return / std_return)


def evaluate_predictions(
    df: pd.DataFrame,
    prob_col: str,
    label_col: str = "builder_hit",
    odds_col: str = "decimal_odds",
    stake_col: str = "stake",
    n_bins: int = 10,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        df: DataFrame with predictions and outcomes
        prob_col: Column name for predicted probabilities
        label_col: Column name for true labels
        odds_col: Column name for decimal odds
        stake_col: Column name for stakes
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with evaluation metrics
    """
    y_true = df[label_col].to_numpy()
    probs = df[prob_col].to_numpy()
    odds = df[odds_col].to_numpy()
    stakes = df[stake_col].to_numpy()

    # Basic metrics
    hit_rate = float(y_true.mean())
    pred_hit_rate = float(probs.mean())
    n_obs = int(len(df))

    # Financial metrics
    roi = compute_roi(y_true, odds, stakes)
    profit = compute_profit(y_true, odds, stakes)
    yield_val = compute_yield(y_true, odds, stakes)
    sharpe = compute_sharpe_ratio(y_true, odds, stakes)

    # Probability metrics
    brier = float(brier_score_loss(y_true, probs))

    # Calibration
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, probs, n_bins=n_bins, strategy="quantile"
        )
        calibration_data = {
            "mean_pred": mean_pred.tolist(),
            "frac_pos": frac_pos.tolist(),
        }
    except Exception as e:
        logger.warning(f"Error computing calibration curve: {e}")
        calibration_data = {"mean_pred": [], "frac_pos": []}

    return {
        "n_obs": n_obs,
        "hit_rate": hit_rate,
        "pred_hit_rate": pred_hit_rate,
        "roi": roi,
        "profit": profit,
        "yield": yield_val,
        "sharpe_ratio": sharpe,
        "brier": brier,
        "total_stake": float(stakes.sum()),
        "calibration": calibration_data,
    }


def evaluate_by_segments(
    df: pd.DataFrame,
    segment_col: str,
    prob_col: str,
    label_col: str = "builder_hit",
    odds_col: str = "decimal_odds",
    stake_col: str = "stake",
) -> pd.DataFrame:
    """
    Evaluate performance across different segments.

    Args:
        df: DataFrame with predictions and outcomes
        segment_col: Column to segment by
        prob_col: Column name for predicted probabilities
        label_col: Column name for true labels
        odds_col: Column name for decimal odds
        stake_col: Column name for stakes

    Returns:
        DataFrame with metrics per segment
    """
    logger.info(f"Evaluating by segment: {segment_col}")

    results = []

    for segment_value, group in df.groupby(segment_col):
        if len(group) == 0:
            continue

        metrics = evaluate_predictions(
            group,
            prob_col=prob_col,
            label_col=label_col,
            odds_col=odds_col,
            stake_col=stake_col,
        )

        metrics[segment_col] = segment_value
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Remove calibration data for cleaner display
    if "calibration" in results_df.columns:
        results_df = results_df.drop(columns=["calibration"])

    logger.info(f"Evaluated {len(results_df)} segments")

    return results_df


def analyze_bet_sizing(
    df: pd.DataFrame,
    prob_col: str,
    label_col: str = "builder_hit",
    odds_col: str = "decimal_odds",
    kelly_fraction: float = 0.25,
) -> Dict:
    """
    Analyze optimal bet sizing strategies.

    Args:
        df: DataFrame with predictions and outcomes
        prob_col: Column name for predicted probabilities
        label_col: Column name for true labels
        odds_col: Column name for decimal odds
        kelly_fraction: Fraction of Kelly criterion to use

    Returns:
        Dictionary with bet sizing analysis
    """
    logger.info("Analyzing bet sizing strategies...")

    probs = df[prob_col].to_numpy()
    odds = df[odds_col].to_numpy()
    y_true = df[label_col].to_numpy()

    # Kelly criterion: f = (bp - q) / b
    # where b = odds - 1, p = win probability, q = 1 - p
    b = odds - 1
    p = probs
    q = 1 - p

    kelly_stakes = (b * p - q) / b
    kelly_stakes = np.maximum(kelly_stakes, 0)  # No negative stakes

    # Fractional Kelly
    fractional_kelly = kelly_stakes * kelly_fraction

    # Compute ROI with different strategies
    strategies = {
        "unit_stake": np.ones_like(probs),
        "full_kelly": kelly_stakes,
        "fractional_kelly": fractional_kelly,
        "probability_weighted": probs,
    }

    results = {}
    for strategy_name, stakes in strategies.items():
        # Normalize stakes to have same total
        if stakes.sum() > 0:
            normalized_stakes = stakes / stakes.sum() * len(stakes)
        else:
            normalized_stakes = stakes

        roi = compute_roi(y_true, odds, normalized_stakes)
        profit = compute_profit(y_true, odds, normalized_stakes)
        sharpe = compute_sharpe_ratio(y_true, odds, normalized_stakes)

        results[strategy_name] = {
            "roi": roi,
            "profit": profit,
            "sharpe": sharpe,
            "avg_stake": float(normalized_stakes.mean()),
            "max_stake": float(normalized_stakes.max()),
        }

        logger.info(
            f"{strategy_name}: ROI={roi*100:.2f}%, "
            f"Profit={profit:.2f}, Sharpe={sharpe:.3f}"
        )

    return results


def compute_drawdown(
    hits: np.ndarray,
    decimal_odds: np.ndarray,
    stakes: np.ndarray | float = 1.0,
) -> Dict:
    """
    Compute drawdown statistics.

    Args:
        hits: Binary array of hits (1) or misses (0)
        decimal_odds: Array of decimal odds
        stakes: Array of stakes or single stake value

    Returns:
        Dictionary with drawdown statistics
    """
    stakes_arr = np.asarray(stakes, dtype=float)
    if stakes_arr.shape == ():
        stakes_arr = np.full_like(decimal_odds, fill_value=stakes_arr, dtype=float)

    # Calculate cumulative profit
    profits = np.where(
        hits == 1,
        stakes_arr * (decimal_odds - 1.0),
        -stakes_arr
    )

    cum_profit = np.cumsum(profits)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_profit)

    # Drawdown at each point
    drawdown = running_max - cum_profit

    # Maximum drawdown
    max_drawdown = float(drawdown.max())

    # Longest drawdown period
    in_drawdown = drawdown > 0
    drawdown_periods = []
    current_period = 0

    for is_down in in_drawdown:
        if is_down:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0

    if current_period > 0:
        drawdown_periods.append(current_period)

    max_drawdown_period = int(max(drawdown_periods)) if drawdown_periods else 0

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_period": max_drawdown_period,
        "final_profit": float(cum_profit[-1]),
        "cum_profit_series": cum_profit.tolist(),
    }
