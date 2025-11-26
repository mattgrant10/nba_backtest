"""Backtesting simulator for bet builder models."""

from __future__ import annotations

from typing import List

import pandas as pd
from tqdm import tqdm

from .evaluation import evaluate_predictions
from .splits import TimeSplitConfig, time_based_splits
from ..config import get_logger, model_config
from ..models.bet_builder_outcome import (
    default_feature_columns,
    train_builder_hit_model,
)


logger = get_logger(__name__)


def run_rolling_backtest(
    builders: pd.DataFrame,
    split_cfg: TimeSplitConfig,
    feature_cols: List[str] | None = None,
    label_col: str = "builder_hit",
    random_state: int | None = None,
    save_predictions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Run a rolling window backtest for the builder hit model.

    For each time split:
        - fit on the training window
        - predict on the test window
        - compute evaluation metrics

    Args:
        builders: DataFrame with builder-level features
        split_cfg: Configuration for time splits
        feature_cols: List of feature columns
        label_col: Target column name
        random_state: Random seed
        save_predictions: Whether to save individual predictions

    Returns:
        Tuple of (results_df, predictions_df)
        - results_df: Per-fold metrics
        - predictions_df: All predictions (if save_predictions=True)
    """
    feature_cols = feature_cols or default_feature_columns()
    random_state = random_state or model_config.random_state

    logger.info("=" * 80)
    logger.info("Starting rolling window backtest")
    logger.info("=" * 80)
    logger.info(f"Features: {', '.join(feature_cols)}")
    logger.info(f"Split config: {split_cfg}")

    df = builders.copy()

    results = []
    all_predictions = [] if save_predictions else None
    fold_id = 0

    for train_idx, test_idx in tqdm(
        time_based_splits(df, split_cfg),
        desc="Backtest folds",
        unit="fold"
    ):
        fold_id += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_id}")
        logger.info(f"{'='*60}")

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        logger.info(
            f"Train: {train_df[split_cfg.date_col].min()} to "
            f"{train_df[split_cfg.date_col].max()} ({len(train_df):,} obs)"
        )
        logger.info(
            f"Test: {test_df[split_cfg.date_col].min()} to "
            f"{test_df[split_cfg.date_col].max()} ({len(test_df):,} obs)"
        )

        # Train model
        try:
            model, train_metrics = train_builder_hit_model(
                train_df,
                feature_cols=feature_cols,
                label_col=label_col,
                random_state=random_state,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error training model in fold {fold_id}: {e}")
            continue

        # Predict on test set
        try:
            X_test = test_df[feature_cols].to_numpy()
            probs_test = model.predict_proba(X_test)[:, 1]

            test_df = test_df.copy()
            test_df["pred_prob"] = probs_test
            test_df["fold"] = fold_id

            # Evaluate
            metrics_test = evaluate_predictions(
                test_df,
                prob_col="pred_prob",
                label_col=label_col,
                odds_col="decimal_odds",
                stake_col="stake",
            )

            # Add fold info
            metrics_test["fold"] = fold_id
            metrics_test["train_n"] = int(len(train_df))
            metrics_test["test_n"] = int(len(test_df))
            metrics_test["train_start"] = str(train_df[split_cfg.date_col].min())
            metrics_test["train_end"] = str(train_df[split_cfg.date_col].max())
            metrics_test["test_start"] = str(test_df[split_cfg.date_col].min())
            metrics_test["test_end"] = str(test_df[split_cfg.date_col].max())

            # Add train metrics
            metrics_test["train_accuracy"] = train_metrics.get("accuracy", None)
            metrics_test["train_roc_auc"] = train_metrics.get("roc_auc", None)

            results.append(metrics_test)

            # Log key metrics
            logger.info(f"Test ROI: {metrics_test['roi']*100:.2f}%")
            logger.info(f"Test Hit Rate: {metrics_test['hit_rate']*100:.2f}%")
            logger.info(f"Test Brier Score: {metrics_test['brier']:.4f}")
            logger.info(f"Test Profit: {metrics_test['profit']:.2f}")

            # Save predictions if requested
            if save_predictions:
                all_predictions.append(
                    test_df[
                        ["builder_id", "game_date", "pred_prob",
                         label_col, "decimal_odds", "stake", "fold"]
                    ]
                )

        except Exception as e:
            logger.error(f"Error evaluating fold {fold_id}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Remove calibration column for cleaner output
    if "calibration" in results_df.columns:
        results_df = results_df.drop(columns=["calibration"])

    logger.info("\n" + "=" * 80)
    logger.info("Backtest Summary")
    logger.info("=" * 80)

    if len(results_df) > 0:
        logger.info(f"Total folds: {len(results_df)}")
        logger.info(f"Average ROI: {results_df['roi'].mean()*100:.2f}%")
        logger.info(f"Median ROI: {results_df['roi'].median()*100:.2f}%")
        logger.info(f"Std ROI: {results_df['roi'].std()*100:.2f}%")
        logger.info(f"Total Profit: {results_df['profit'].sum():.2f}")
        logger.info(f"Average Hit Rate: {results_df['hit_rate'].mean()*100:.2f}%")
        logger.info(f"Average Brier: {results_df['brier'].mean():.4f}")
        logger.info(f"Win Rate (positive ROI): {(results_df['roi'] > 0).mean()*100:.1f}%")
    else:
        logger.warning("No results generated")

    # Combine predictions if saved
    predictions_df = None
    if save_predictions and all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"Saved {len(predictions_df):,} predictions")

    return results_df, predictions_df


def run_expanding_backtest(
    builders: pd.DataFrame,
    split_cfg: TimeSplitConfig,
    feature_cols: List[str] | None = None,
    label_col: str = "builder_hit",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Run an expanding window backtest.

    Similar to rolling backtest but training window expands over time
    instead of sliding.

    Args:
        builders: DataFrame with builder-level features
        split_cfg: Configuration for time splits
        feature_cols: List of feature columns
        label_col: Target column name
        random_state: Random seed

    Returns:
        DataFrame with per-fold metrics
    """
    from .splits import expanding_window_splits

    feature_cols = feature_cols or default_feature_columns()
    random_state = random_state or model_config.random_state

    logger.info("=" * 80)
    logger.info("Starting expanding window backtest")
    logger.info("=" * 80)

    df = builders.copy()
    results = []
    fold_id = 0

    for train_idx, test_idx in tqdm(
        expanding_window_splits(df, split_cfg),
        desc="Backtest folds",
        unit="fold"
    ):
        fold_id += 1

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        logger.info(
            f"Fold {fold_id}: Train size={len(train_df):,}, "
            f"Test size={len(test_df):,}"
        )

        # Train and evaluate
        try:
            model, _ = train_builder_hit_model(
                train_df,
                feature_cols=feature_cols,
                label_col=label_col,
                random_state=random_state,
                verbose=False,
            )

            X_test = test_df[feature_cols].to_numpy()
            probs_test = model.predict_proba(X_test)[:, 1]

            test_df = test_df.copy()
            test_df["pred_prob"] = probs_test

            metrics_test = evaluate_predictions(
                test_df,
                prob_col="pred_prob",
                label_col=label_col,
                odds_col="decimal_odds",
                stake_col="stake",
            )

            metrics_test["fold"] = fold_id
            metrics_test["train_n"] = int(len(train_df))
            metrics_test["test_n"] = int(len(test_df))

            results.append(metrics_test)

        except Exception as e:
            logger.error(f"Error in fold {fold_id}: {e}")
            continue

    results_df = pd.DataFrame(results)

    if "calibration" in results_df.columns:
        results_df = results_df.drop(columns=["calibration"])

    logger.info(f"\nCompleted {len(results_df)} folds")
    if len(results_df) > 0:
        logger.info(f"Average ROI: {results_df['roi'].mean()*100:.2f}%")

    return results_df


def analyze_backtest_results(results_df: pd.DataFrame) -> dict:
    """
    Perform detailed analysis of backtest results.

    Args:
        results_df: DataFrame with backtest results

    Returns:
        Dictionary with analysis
    """
    logger.info("Analyzing backtest results...")

    analysis = {}

    # Overall statistics
    analysis["overall"] = {
        "n_folds": len(results_df),
        "mean_roi": float(results_df["roi"].mean()),
        "median_roi": float(results_df["roi"].median()),
        "std_roi": float(results_df["roi"].std()),
        "min_roi": float(results_df["roi"].min()),
        "max_roi": float(results_df["roi"].max()),
        "total_profit": float(results_df["profit"].sum()),
        "total_stake": float(results_df["total_stake"].sum()),
        "overall_roi": float(
            results_df["profit"].sum() / results_df["total_stake"].sum()
        ),
        "win_rate": float((results_df["roi"] > 0).mean()),
    }

    logger.info("Overall statistics:")
    for key, value in analysis["overall"].items():
        if isinstance(value, float) and "roi" in key.lower():
            logger.info(f"  {key}: {value*100:.2f}%")
        else:
            logger.info(f"  {key}: {value:.4f}")

    # Time-based trends
    if "test_start" in results_df.columns:
        results_df["test_date"] = pd.to_datetime(results_df["test_start"])
        results_df = results_df.sort_values("test_date")

        # Check for trends
        from scipy.stats import linregress

        x = np.arange(len(results_df))
        y = results_df["roi"].values

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        analysis["trend"] = {
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

        trend_direction = "improving" if slope > 0 else "declining"
        logger.info(f"ROI trend: {trend_direction} (slope={slope:.6f}, p={p_value:.4f})")

    return analysis
