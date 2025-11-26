"""Configuration module for bet builder research project."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_artifacts"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""

    date_col: str = "game_date"
    train_window_days: int = 60
    test_window_days: int = 7
    min_train_size: int = 500
    min_games_per_player: int = 20
    random_state: int = 42
    stake_col: str = "stake"
    odds_col: str = "decimal_odds"
    label_col: str = "builder_hit"


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    stat_cols: List[str] = field(default_factory=lambda: ["pts", "fg3m", "ast", "reb"])
    min_games: int = 20
    correlation_method: str = "spearman"
    random_state: int = 42

    # Feature columns for builder outcome model
    feature_cols: List[str] = field(default_factory=lambda: [
        "n_legs",
        "mean_abs_edge",
        "max_abs_edge",
        "book_implied_prob",
    ])


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


# Global configs
backtest_config = BacktestConfig()
model_config = ModelConfig()
logging_config = LoggingConfig()


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional file path to write logs to
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=logging_config.format,
        datefmt=logging_config.date_format,
        handlers=handlers,
        force=True
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
