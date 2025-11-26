#!/usr/bin/env python3
"""
Fit statistical distributions to player performance data.

This script analyzes player stat distributions and fits parametric models
to estimate probabilities for prop bets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, setup_logging, get_logger, model_config
from src.models.stat_distributions import (
    attach_parametric_models,
    summarise_player_stat_distributions,
    compute_tail_probabilities,
    compute_value_bets,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit player stat distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "player_games.parquet",
        help="Input player games file",
    )

    parser.add_argument(
        "--prop-lines-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "player_games.parquet",  # Will try raw data if not in processed
        help="Prop lines file (optional, for tail probability calculation)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for models",
    )

    parser.add_argument(
        "--min-games",
        type=int,
        default=model_config.min_games,
        help="Minimum games required per player",
    )

    parser.add_argument(
        "--compute-tail-probs",
        action="store_true",
        help="Compute tail probabilities for prop lines",
    )

    parser.add_argument(
        "--find-value-bets",
        action="store_true",
        help="Identify potential value bets",
    )

    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge for value bet identification",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(__import__("logging"), log_level), log_file=args.log_file)

    logger.info("=" * 80)
    logger.info("PLAYER STAT DISTRIBUTION FITTING")
    logger.info("=" * 80)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load player games
        logger.info(f"\nLoading player games from {args.input_file}...")
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        player_games = pd.read_parquet(args.input_file)
        logger.info(f"Loaded {len(player_games):,} game logs")

        # Summarize distributions
        logger.info(f"\nComputing distribution summaries (min_games={args.min_games})...")
        summary = summarise_player_stat_distributions(
            player_games,
            stat_cols=model_config.stat_cols,
            min_games=args.min_games
        )

        # Fit parametric models
        logger.info("\nFitting parametric models...")
        summary_with_models = attach_parametric_models(summary)

        # Save models
        out_path = args.output_dir / "player_stat_distributions.parquet"
        summary_with_models.to_parquet(out_path, index=False)
        logger.info(f" Saved distribution models to {out_path}")

        # Compute tail probabilities if requested
        if args.compute_tail_probs:
            from src.data_loader import load_prop_lines

            logger.info("\nComputing tail probabilities...")
            try:
                prop_lines = load_prop_lines()

                tail_probs = compute_tail_probabilities(
                    summary_with_models,
                    prop_lines
                )

                tail_probs_path = args.output_dir / "tail_probabilities.parquet"
                tail_probs.to_parquet(tail_probs_path, index=False)
                logger.info(f" Saved tail probabilities to {tail_probs_path}")

                # Find value bets if requested
                if args.find_value_bets:
                    logger.info(f"\nIdentifying value bets (min_edge={args.min_edge})...")
                    value_bets = compute_value_bets(
                        tail_probs,
                        prop_lines,
                        min_edge=args.min_edge
                    )

                    value_bets_path = args.output_dir / "value_bets.parquet"
                    value_bets.to_parquet(value_bets_path, index=False)
                    logger.info(f" Saved {len(value_bets):,} value bets to {value_bets_path}")

            except Exception as e:
                logger.warning(f"Could not compute tail probabilities: {e}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Players analyzed: {summary_with_models['player_id'].nunique():,}")
        logger.info(f"Total player-stat combinations: {len(summary_with_models):,}")

        # Distribution of model families
        family_dist = summary_with_models["family"].value_counts()
        logger.info("\nModel families:")
        for family, count in family_dist.items():
            pct = count / len(summary_with_models) * 100
            logger.info(f"  {family}: {count:,} ({pct:.1f}%)")

        logger.info("\n Distribution fitting completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during distribution fitting: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
