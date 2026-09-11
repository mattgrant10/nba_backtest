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
OUTPUT_DIR = PROCESSED_DATA_DIR
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
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


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_simulations: int = 10000
    random_state: int = 42
    use_gpu: bool = True  # Use Apple Metal GPU via MLX if available
    use_correlations: bool = True  # Apply correlation structure between stats
    batch_size: int = 5000  # Batch size for GPU processing
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )


@dataclass
class LineGeneratorConfig:
    """
    Configuration for learned line generator model.

    Sample Size Thresholds:
        - min_samples_train (500): Minimum to start training, but high variance
        - min_samples_reliable (2000): Reasonable for cross-validation
        - min_samples_per_stat (100): Per stat type minimum for reliable patterns
        - Optimal: 5000+ samples for production-ready model
    """

    # Model type: "gradient_boosting" (default), "linear" (fallback for small samples)
    model_type: str = "gradient_boosting"
    random_state: int = 42

    # Feature computation windows
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Sample size thresholds
    min_samples_train: int = 500       # Minimum to start training
    min_samples_reliable: int = 2000   # Minimum for reliable simulation
    min_samples_per_stat: int = 100    # Per stat type minimum

    # Line rounding
    rounding_increment: float = 0.5    # Standard betting line increment

    # Stats to model
    stat_cols: List[str] = field(default_factory=lambda: ["pts", "fg3m", "ast", "reb"])

    # Gradient Boosting hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1

    # Default odds range (used when odds not learned)
    default_odds_min: float = 1.85
    default_odds_max: float = 1.95


# Global configs
backtest_config = BacktestConfig()
model_config = ModelConfig()
logging_config = LoggingConfig()
monte_carlo_config = MonteCarloConfig()
line_generator_config = LineGeneratorConfig()


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
