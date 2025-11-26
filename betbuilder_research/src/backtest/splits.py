"""Time-based data splitting for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from ..config import get_logger


logger = get_logger(__name__)


@dataclass
class TimeSplitConfig:
    """Configuration for time-based splits."""

    date_col: str = "game_date"
    train_window_days: int = 60
    test_window_days: int = 7
    min_train_size: int = 200
    step_days: int | None = None  # If None, uses test_window_days

    def __post_init__(self):
        """Validate configuration."""
        if self.train_window_days <= 0:
            raise ValueError("train_window_days must be positive")
        if self.test_window_days <= 0:
            raise ValueError("test_window_days must be positive")
        if self.min_train_size < 0:
            raise ValueError("min_train_size must be non-negative")

        if self.step_days is None:
            self.step_days = self.test_window_days


def time_based_splits(
    df: pd.DataFrame,
    cfg: TimeSplitConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield rolling-origin time-based train/test index splits.

    This creates a rolling window where:
    - Training window: cfg.train_window_days
    - Test window: cfg.test_window_days
    - Window moves forward by cfg.step_days after each fold

    Args:
        df: DataFrame with date column
        cfg: Split configuration

    Yields:
        Tuples of (train_idx, test_idx) as integer index arrays
    """
    logger.info(
        f"Creating time-based splits: "
        f"train={cfg.train_window_days}d, test={cfg.test_window_days}d, "
        f"step={cfg.step_days}d"
    )

    # Sort by date and preserve original index
    df_sorted = df.sort_values(cfg.date_col).reset_index(drop=False)

    # Ensure date column is datetime.date
    if not isinstance(df_sorted[cfg.date_col].iloc[0], pd.Timestamp):
        all_dates = pd.to_datetime(df_sorted[cfg.date_col]).dt.date
    else:
        all_dates = df_sorted[cfg.date_col].dt.date

    unique_dates = sorted(all_dates.unique())

    if not unique_dates:
        logger.warning("No dates found in dataset")
        return

    first_date = unique_dates[0]
    last_date = unique_dates[-1]

    logger.info(
        f"Date range: {first_date} to {last_date} "
        f"({(last_date - first_date).days} days)"
    )

    n_folds = 0
    current_start = first_date

    while True:
        train_end = current_start + timedelta(days=cfg.train_window_days)
        test_end = train_end + timedelta(days=cfg.test_window_days)

        # Check if we have enough data for this fold
        if test_end > last_date:
            break

        # Create masks
        train_mask = (all_dates >= current_start) & (all_dates < train_end)
        test_mask = (all_dates >= train_end) & (all_dates < test_end)

        # Get original indices
        train_idx = df_sorted.loc[train_mask, "index"].to_numpy()
        test_idx = df_sorted.loc[test_mask, "index"].to_numpy()

        # Validate minimum sizes
        if len(train_idx) >= cfg.min_train_size and len(test_idx) > 0:
            n_folds += 1
            logger.debug(
                f"Fold {n_folds}: train {current_start} to {train_end} "
                f"({len(train_idx)} obs), "
                f"test {train_end} to {test_end} ({len(test_idx)} obs)"
            )
            yield train_idx, test_idx
        else:
            logger.debug(
                f"Skipping fold: train_size={len(train_idx)}, "
                f"test_size={len(test_idx)}"
            )

        # Move window forward
        current_start = current_start + timedelta(days=cfg.step_days)

    logger.info(f"Created {n_folds} time-based folds")


def expanding_window_splits(
    df: pd.DataFrame,
    cfg: TimeSplitConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield expanding window time-based splits.

    Unlike rolling windows, the training set grows over time,
    always starting from the first date.

    Args:
        df: DataFrame with date column
        cfg: Split configuration

    Yields:
        Tuples of (train_idx, test_idx) as integer index arrays
    """
    logger.info(
        f"Creating expanding window splits: "
        f"initial_train={cfg.train_window_days}d, test={cfg.test_window_days}d"
    )

    df_sorted = df.sort_values(cfg.date_col).reset_index(drop=False)

    if not isinstance(df_sorted[cfg.date_col].iloc[0], pd.Timestamp):
        all_dates = pd.to_datetime(df_sorted[cfg.date_col]).dt.date
    else:
        all_dates = df_sorted[cfg.date_col].dt.date

    unique_dates = sorted(all_dates.unique())

    if not unique_dates:
        logger.warning("No dates found in dataset")
        return

    first_date = unique_dates[0]
    last_date = unique_dates[-1]

    logger.info(f"Date range: {first_date} to {last_date}")

    n_folds = 0
    train_end = first_date + timedelta(days=cfg.train_window_days)

    while True:
        test_end = train_end + timedelta(days=cfg.test_window_days)

        if test_end > last_date:
            break

        # Training set always starts from first_date
        train_mask = (all_dates >= first_date) & (all_dates < train_end)
        test_mask = (all_dates >= train_end) & (all_dates < test_end)

        train_idx = df_sorted.loc[train_mask, "index"].to_numpy()
        test_idx = df_sorted.loc[test_mask, "index"].to_numpy()

        if len(train_idx) >= cfg.min_train_size and len(test_idx) > 0:
            n_folds += 1
            logger.debug(
                f"Fold {n_folds}: train {first_date} to {train_end} "
                f"({len(train_idx)} obs), "
                f"test {train_end} to {test_end} ({len(test_idx)} obs)"
            )
            yield train_idx, test_idx

        # Move test window forward
        train_end = train_end + timedelta(days=cfg.step_days)

    logger.info(f"Created {n_folds} expanding window folds")


def get_split_summary(
    df: pd.DataFrame,
    cfg: TimeSplitConfig,
    split_type: str = "rolling"
) -> pd.DataFrame:
    """
    Generate a summary of splits without actually creating them.

    Args:
        df: DataFrame with date column
        cfg: Split configuration
        split_type: 'rolling' or 'expanding'

    Returns:
        DataFrame with split information
    """
    logger.info(f"Generating {split_type} split summary...")

    if split_type == "rolling":
        splits_gen = time_based_splits(df, cfg)
    elif split_type == "expanding":
        splits_gen = expanding_window_splits(df, cfg)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    summaries = []
    for fold_id, (train_idx, test_idx) in enumerate(splits_gen):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        summaries.append({
            "fold": fold_id,
            "train_start": train_df[cfg.date_col].min(),
            "train_end": train_df[cfg.date_col].max(),
            "test_start": test_df[cfg.date_col].min(),
            "test_end": test_df[cfg.date_col].max(),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })

    summary_df = pd.DataFrame(summaries)

    logger.info(f"Split summary generated with {len(summary_df)} folds")
    if len(summary_df) > 0:
        logger.info(f"Average train size: {summary_df['train_size'].mean():.0f}")
        logger.info(f"Average test size: {summary_df['test_size'].mean():.0f}")

    return summary_df
