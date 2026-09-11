#!/usr/bin/env python3
"""
Train the Learned Line Generator model.

This script trains a machine learning model to learn how bookmakers set
prop lines based on real betting line data, then saves the model for
later use in synthetic line generation.

Usage:
    python scripts/train_line_generator.py --lines-file data/raw/real_betting_lines.csv

    # With custom output directory
    python scripts/train_line_generator.py \
        --lines-file data/raw/real_betting_lines.csv \
        --output-dir models_artifacts

    # Force linear model (for small samples)
    python scripts/train_line_generator.py \
        --lines-file data/raw/real_betting_lines.csv \
        --model-type linear

Expected CSV Format:
    Required columns: player_id, stat, line, game_date
    Optional columns: over_odds, under_odds, book

Sample Size Thresholds:
    - Minimum to train: 500 samples
    - Reliable simulation: 2,000+ samples
    - Optimal: 5,000+ samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    setup_logging,
    get_logger,
    MODELS_DIR,
    LineGeneratorConfig,
    line_generator_config,
)
from src.data_loader import load_player_game_logs
from src.models.line_generator import (
    LearnedLineGenerator,
    load_real_lines_csv,
    validate_real_lines_csv,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Learned Line Generator model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--lines-file",
        type=Path,
        required=True,
        help="Path to CSV file with real betting lines"
    )

    parser.add_argument(
        "--player-games-file",
        type=Path,
        default=None,
        help="Optional path to player game logs (auto-detected if not provided)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Directory to save trained model (default: {MODELS_DIR})"
    )

    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "linear"],
        default="gradient_boosting",
        help="Model type to use (default: gradient_boosting)"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum samples required to train (default: 500)"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the CSV file without training"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging()

    logger.info("=" * 60)
    logger.info("LEARNED LINE GENERATOR - TRAINING")
    logger.info("=" * 60)

    # Validate input file
    logger.info(f"Validating input file: {args.lines_file}")
    is_valid, errors = validate_real_lines_csv(args.lines_file)

    if not is_valid:
        logger.error("Validation failed:")
        for err in errors:
            logger.error(f"  - {err}")
        return 1

    if args.validate_only:
        logger.info("Validation passed. Exiting (--validate-only)")
        return 0

    # Load data
    logger.info("Loading real betting lines...")
    try:
        real_lines = load_real_lines_csv(args.lines_file)
    except Exception as e:
        logger.error(f"Error loading lines: {e}")
        return 1

    n_samples = len(real_lines)
    logger.info(f"Loaded {n_samples:,} betting lines")

    # Check sample size threshold
    if n_samples < args.min_samples:
        logger.error(
            f"Insufficient samples: {n_samples} < {args.min_samples} minimum. "
            f"Collect more data or reduce --min-samples threshold."
        )
        return 1

    # Load player games
    logger.info("Loading player game logs...")
    try:
        if args.player_games_file:
            player_games = load_player_game_logs(args.player_games_file)
        else:
            player_games = load_player_game_logs()
    except Exception as e:
        logger.error(f"Error loading player games: {e}")
        return 1

    logger.info(f"Loaded {len(player_games):,} player game records")

    # Configure model
    config = LineGeneratorConfig(
        model_type=args.model_type,
        min_samples_train=args.min_samples,
    )

    # Train model
    logger.info("-" * 40)
    logger.info("Training model...")
    logger.info("-" * 40)

    try:
        generator = LearnedLineGenerator(config=config)
        metrics = generator.fit(
            player_games=player_games,
            real_lines=real_lines,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Report results
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Samples used: {metrics.n_samples:,}")
    logger.info(f"Model type: {metrics.model_type}")
    logger.info(f"Confidence level: {metrics.confidence_level}")
    logger.info(f"Is reliable: {metrics.is_reliable}")
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  MAE (Mean Absolute Error): {metrics.mae:.3f}")
    logger.info(f"  RMSE: {metrics.rmse:.3f}")
    logger.info(f"  R-squared: {metrics.r2:.3f}")
    logger.info("")
    logger.info("Per-Stat MAE:")
    for stat, mae in metrics.mae_by_stat.items():
        samples = metrics.samples_by_stat.get(stat, 0)
        logger.info(f"  {stat}: {mae:.3f} ({samples:,} samples)")

    # Confidence interpretation
    logger.info("")
    if metrics.confidence_level == "high":
        logger.info("HIGH CONFIDENCE: Model is production-ready.")
    elif metrics.confidence_level == "medium":
        logger.info("MEDIUM CONFIDENCE: Model is reasonably reliable.")
    elif metrics.confidence_level == "low":
        logger.info(
            "LOW CONFIDENCE: Model trained but may have high variance. "
            "Consider collecting more data (2000+ samples recommended)."
        )
    else:
        logger.warning("INSUFFICIENT DATA: Model may not be reliable.")

    # Save model
    output_path = args.output_dir / "line_generator.joblib"
    logger.info("")
    logger.info(f"Saving model to: {output_path}")

    try:
        generator.save(output_path)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return 1

    logger.info("")
    logger.info("SUCCESS! Model saved successfully.")
    logger.info("")
    logger.info("To use the trained model for line generation:")
    logger.info(f"  from src.models import LearnedLineGenerator")
    logger.info(f"  model = LearnedLineGenerator.load('{output_path}')")
    logger.info(f"  lines = model.generate_lines(player_games)")
    logger.info("")
    logger.info("Or via load_prop_lines:")
    logger.info(f"  from src.data_loader import load_prop_lines")
    logger.info(f"  lines = load_prop_lines(learned_model_path='{output_path}')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
