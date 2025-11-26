#!/usr/bin/env python3
"""
Analyze correlations and dependencies between player stats.

This script computes per-player and global correlations to understand
how different stats relate to each other.
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
from src.models.dependencies import (
    compute_global_correlation,
    compute_player_stat_correlations,
    analyze_correlation_patterns,
    identify_high_correlation_pairs,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze stat correlations and dependencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "player_games.parquet",
        help="Input player games file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for correlation data",
    )

    parser.add_argument(
        "--min-games",
        type=int,
        default=model_config.min_games,
        help="Minimum games required per player",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=model_config.correlation_method,
        choices=["spearman", "pearson"],
        help="Correlation method",
    )

    parser.add_argument(
        "--high-corr-threshold",
        type=float,
        default=0.7,
        help="Threshold for identifying high correlations",
    )

    parser.add_argument(
        "--analyze-patterns",
        action="store_true",
        default=True,
        help="Perform detailed correlation pattern analysis",
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
    logger.info("STAT CORRELATION ANALYSIS")
    logger.info("=" * 80)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load player games
        logger.info(f"\nLoading player games from {args.input_file}...")
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        player_games = pd.read_parquet(args.input_file)
        logger.info(f"Loaded {len(player_games):,} game logs")

        # Compute per-player correlations
        logger.info(f"\nComputing per-player correlations ({args.method} method)...")
        per_player_corr = compute_player_stat_correlations(
            player_games,
            stat_cols=model_config.stat_cols,
            min_games=args.min_games,
            method=args.method
        )

        per_player_path = args.output_dir / "player_stat_correlations.parquet"
        per_player_corr.to_parquet(per_player_path, index=False)
        logger.info(f" Saved per-player correlations to {per_player_path}")

        # Compute global correlations
        logger.info("\nComputing global correlations...")
        global_corr = compute_global_correlation(
            player_games,
            stat_cols=model_config.stat_cols,
            method=args.method
        )

        global_path = args.output_dir / "global_stat_correlations.parquet"
        global_corr.to_parquet(global_path, index=False)
        logger.info(f" Saved global correlations to {global_path}")

        # Analyze patterns
        if args.analyze_patterns:
            logger.info("\nAnalyzing correlation patterns...")
            patterns = analyze_correlation_patterns(per_player_corr, global_corr)

            if "per_player_summary" in patterns:
                summary_path = args.output_dir / "correlation_summary.parquet"
                patterns["per_player_summary"].to_parquet(summary_path, index=False)
                logger.info(f" Saved correlation summary to {summary_path}")

            if "comparison" in patterns:
                comparison_path = args.output_dir / "correlation_comparison.parquet"
                patterns["comparison"].to_parquet(comparison_path, index=False)
                logger.info(f" Saved correlation comparison to {comparison_path}")

        # Identify high correlation pairs
        logger.info(f"\nIdentifying high correlations (threshold={args.high_corr_threshold})...")
        high_corr = identify_high_correlation_pairs(
            per_player_corr,
            threshold=args.high_corr_threshold
        )

        if len(high_corr) > 0:
            high_corr_path = args.output_dir / "high_correlations.parquet"
            high_corr.to_parquet(high_corr_path, index=False)
            logger.info(f" Saved {len(high_corr):,} high correlations to {high_corr_path}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Players analyzed: {per_player_corr['player_id'].nunique():,}")
        logger.info(f"Total correlations computed: {len(per_player_corr):,}")
        logger.info(f"High correlations found: {len(high_corr):,}")

        # Show strongest global correlations
        logger.info("\nStrongest global correlations:")
        global_off_diag = global_corr[global_corr["stat_x"] != global_corr["stat_y"]]
        for _, row in global_off_diag.nlargest(5, "corr").iterrows():
            logger.info(f"  {row['stat_x']} <-> {row['stat_y']}: {row['corr']:.3f}")

        logger.info("\n Correlation analysis completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during correlation analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
