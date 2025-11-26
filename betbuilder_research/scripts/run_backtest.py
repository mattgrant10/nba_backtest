#!/usr/bin/env python3
"""
Run bet builder backtesting simulation.

This script performs time-based backtesting of bet builder prediction models,
evaluating performance across multiple time periods.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.backtest.simulator import run_rolling_backtest, analyze_backtest_results
from src.backtest.splits import TimeSplitConfig
from src.config import PROCESSED_DATA_DIR, setup_logging, get_logger, backtest_config, model_config


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run bet builder backtesting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "bet_builders.parquet",
        help="Input bet builders file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for results",
    )

    parser.add_argument(
        "--train-window",
        type=int,
        default=backtest_config.train_window_days,
        help="Training window size in days",
    )

    parser.add_argument(
        "--test-window",
        type=int,
        default=backtest_config.test_window_days,
        help="Test window size in days",
    )

    parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="Step size in days (default: same as test window)",
    )

    parser.add_argument(
        "--min-train-size",
        type=int,
        default=backtest_config.min_train_size,
        help="Minimum training set size",
    )

    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=None,
        help="Feature columns to use (default: use config defaults)",
    )

    parser.add_argument(
        "--label",
        type=str,
        default=backtest_config.label_col,
        help="Label column name",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=model_config.random_state,
        help="Random state for reproducibility",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions from each fold",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        default=True,
        help="Perform detailed result analysis",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="backtest_results",
        help="Base name for output files",
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
    logger.info("BET BUILDER BACKTESTING")
    logger.info("=" * 80)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load builders
        logger.info(f"\nLoading builders from {args.input_file}...")
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        builders = pd.read_parquet(args.input_file)
        logger.info(f"Loaded {len(builders):,} builders")

        # Date range
        date_range = (
            builders[backtest_config.date_col].min(),
            builders[backtest_config.date_col].max()
        )
        logger.info(f"Date range: {date_range[0]} to {date_range[1]}")

        # Configure splits
        split_cfg = TimeSplitConfig(
            date_col=backtest_config.date_col,
            train_window_days=args.train_window,
            test_window_days=args.test_window,
            min_train_size=args.min_train_size,
            step_days=args.step_days,
        )

        logger.info(f"\nBacktest configuration:")
        logger.info(f"  Train window: {split_cfg.train_window_days} days")
        logger.info(f"  Test window: {split_cfg.test_window_days} days")
        logger.info(f"  Step size: {split_cfg.step_days} days")
        logger.info(f"  Min train size: {split_cfg.min_train_size}")

        # Determine features
        from src.models.bet_builder_outcome import default_feature_columns
        feature_cols = args.features if args.features else default_feature_columns()
        logger.info(f"\nUsing {len(feature_cols)} features")

        # Run backtest
        logger.info("\nStarting backtest...\n")
        results_df, predictions_df = run_rolling_backtest(
            builders,
            split_cfg=split_cfg,
            feature_cols=feature_cols,
            label_col=args.label,
            random_state=args.random_state,
            save_predictions=args.save_predictions,
        )

        # Save results
        results_path = args.output_dir / f"{args.output_name}.parquet"
        results_df.to_parquet(results_path, index=False)
        logger.info(f"\n Saved backtest results to {results_path}")

        # Save predictions if requested
        if args.save_predictions and predictions_df is not None:
            predictions_path = args.output_dir / f"{args.output_name}_predictions.parquet"
            predictions_df.to_parquet(predictions_path, index=False)
            logger.info(f" Saved predictions to {predictions_path}")

        # Analyze results
        if args.analyze and len(results_df) > 0:
            logger.info("\n" + "=" * 80)
            logger.info("DETAILED ANALYSIS")
            logger.info("=" * 80)

            analysis = analyze_backtest_results(results_df)

            # Save analysis
            analysis_path = args.output_dir / f"{args.output_name}_analysis.txt"
            with open(analysis_path, "w") as f:
                f.write("BACKTEST ANALYSIS\n")
                f.write("=" * 80 + "\n\n")

                f.write("OVERALL STATISTICS\n")
                f.write("-" * 80 + "\n")
                for key, value in analysis["overall"].items():
                    f.write(f"{key}: {value}\n")

                if "trend" in analysis:
                    f.write("\n\nTIME TREND ANALYSIS\n")
                    f.write("-" * 80 + "\n")
                    for key, value in analysis["trend"].items():
                        f.write(f"{key}: {value}\n")

            logger.info(f" Saved analysis to {analysis_path}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)

        if len(results_df) > 0:
            logger.info(f"Total folds completed: {len(results_df)}")
            logger.info(f"Average ROI: {results_df['roi'].mean()*100:.2f}%")
            logger.info(f"Total profit: {results_df['profit'].sum():.2f}")
            logger.info(f"Total stake: {results_df['total_stake'].sum():.2f}")

            overall_roi = results_df['profit'].sum() / results_df['total_stake'].sum()
            logger.info(f"Overall ROI: {overall_roi*100:.2f}%")

            # Profitability stats
            profitable_folds = (results_df['roi'] > 0).sum()
            logger.info(f"Profitable folds: {profitable_folds}/{len(results_df)} ({profitable_folds/len(results_df)*100:.1f}%)")

            # Best and worst folds
            best_fold = results_df.loc[results_df['roi'].idxmax()]
            worst_fold = results_df.loc[results_df['roi'].idxmin()]

            logger.info(f"\nBest fold:")
            logger.info(f"  Period: {best_fold['test_start']} to {best_fold['test_end']}")
            logger.info(f"  ROI: {best_fold['roi']*100:.2f}%")

            logger.info(f"\nWorst fold:")
            logger.info(f"  Period: {worst_fold['test_start']} to {worst_fold['test_end']}")
            logger.info(f"  ROI: {worst_fold['roi']*100:.2f}%")

        logger.info("\n Backtesting completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during backtesting: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
