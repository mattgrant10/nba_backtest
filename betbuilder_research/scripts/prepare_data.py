#!/usr/bin/env python3
"""
Prepare and validate data for bet builder analysis.

This script loads raw data, performs validation, adds contextual features,
and creates processed datasets ready for modeling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, setup_logging, get_logger
from src.data_loader import (
    load_bet_builder_legs,
    load_player_game_logs,
    load_prop_lines,
    validate_data_consistency,
)
from src.feature_engineering import (
    build_builder_level_dataset,
    build_leg_level_dataset,
    add_stat_composition_features,
    add_player_diversity_features,
)
from src.preprocessing import add_basic_context_features
from src.visualization import (
    display_dataframe,
    display_summary_stats,
    plot_distributions,
    plot_correlation_matrix,
    plot_value_counts,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare and validate bet builder data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data consistency validation",
    )

    parser.add_argument(
        "--add-features",
        action="store_true",
        default=True,
        help="Add additional features (stat composition, player diversity)",
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

    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=True,
        help="Display plots interactively (default: True)",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to disk",
    )

    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: output_dir/plots)",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(__import__("logging"), log_level), log_file=args.log_file)

    logger.info("=" * 80)
    logger.info("BET BUILDER DATA PREPARATION")
    logger.info("=" * 80)

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")

        # Setup plots directory
        plots_dir = args.plots_dir if args.plots_dir else args.output_dir / "plots"
        if args.save_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Plots directory: {plots_dir}")

        # Load raw data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOADING RAW DATA")
        logger.info("=" * 80)

        logger.info("\n>>> Loading player game logs...")
        player_games = load_player_game_logs()
        display_dataframe(player_games, "Player Game Logs - Loaded", max_rows=10)
        display_summary_stats(player_games, title="Player Game Logs - Summary Statistics")

        logger.info("\n>>> Loading prop lines...")
        prop_lines = load_prop_lines()
        display_dataframe(prop_lines, "Prop Lines - Loaded", max_rows=10)
        display_summary_stats(prop_lines, title="Prop Lines - Summary Statistics")

        logger.info("\n>>> Loading bet builder legs...")
        bet_builder_legs = load_bet_builder_legs()
        display_dataframe(bet_builder_legs, "Bet Builder Legs - Loaded", max_rows=10)
        display_summary_stats(bet_builder_legs, title="Bet Builder Legs - Summary Statistics")

        # Visualize distributions of key stats
        logger.info("\n>>> Plotting stat distributions...")
        stat_cols = ["pts", "fg3m", "ast", "reb"]
        plot_distributions(
            player_games,
            columns=stat_cols,
            save_path=plots_dir / "player_stats_distributions.png" if args.save_plots else None,
            show=args.show_plots
        )

        # Plot correlation matrix
        logger.info("\n>>> Plotting stat correlations...")
        plot_correlation_matrix(
            player_games,
            columns=stat_cols,
            save_path=plots_dir / "stat_correlations.png" if args.save_plots else None,
            show=args.show_plots
        )

        # Validate data consistency
        if args.validate:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: VALIDATING DATA CONSISTENCY")
            logger.info("=" * 80)
            validation_results = validate_data_consistency(
                player_games, prop_lines, bet_builder_legs
            )
            logger.info(f"\nValidation stats:")
            for key, value in validation_results['stats'].items():
                logger.info(f"  {key}: {value}")

        # Add contextual features
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: ADDING CONTEXTUAL FEATURES")
        logger.info("=" * 80)

        logger.info("\n>>> Adding basic context features (home/away, rest days)...")
        player_games = add_basic_context_features(player_games)
        display_dataframe(player_games, "Player Game Logs - After Context Features", max_rows=10)

        # Show new features
        if 'is_home' in player_games.columns:
            logger.info(f"\n>>> Home game distribution:")
            plot_value_counts(
                player_games,
                'is_home',
                save_path=plots_dir / "home_away_distribution.png" if args.save_plots else None,
                show=args.show_plots
            )

        if 'days_rest' in player_games.columns:
            logger.info(f"\n>>> Days rest distribution:")
            plot_distributions(
                player_games,
                columns=['days_rest'],
                save_path=plots_dir / "days_rest_distribution.png" if args.save_plots else None,
                show=args.show_plots
            )

        # Build leg-level dataset
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: BUILDING LEG-LEVEL DATASET")
        logger.info("=" * 80)

        logger.info("\n>>> Merging game logs with bet builder legs...")
        leg_df = build_leg_level_dataset(player_games, prop_lines, bet_builder_legs)
        display_dataframe(leg_df, "Leg-Level Dataset", max_rows=15)
        display_summary_stats(leg_df, title="Leg-Level Dataset - Summary Statistics")

        # Visualize leg outcomes
        logger.info("\n>>> Plotting leg hit distribution...")
        plot_value_counts(
            leg_df,
            'leg_hit',
            save_path=plots_dir / "leg_hit_distribution.png" if args.save_plots else None,
            show=args.show_plots
        )

        logger.info("\n>>> Plotting direction distribution...")
        plot_value_counts(
            leg_df,
            'direction_lower',
            save_path=plots_dir / "direction_distribution.png" if args.save_plots else None,
            show=args.show_plots
        )

        # Build builder-level dataset
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: BUILDING BUILDER-LEVEL DATASET")
        logger.info("=" * 80)

        logger.info("\n>>> Aggregating legs to builder level...")
        builder_df = build_builder_level_dataset(leg_df)
        display_dataframe(builder_df, "Builder-Level Dataset", max_rows=15)
        display_summary_stats(builder_df, title="Builder-Level Dataset - Summary Statistics")

        # Visualize builder characteristics
        logger.info("\n>>> Plotting builder characteristics...")
        builder_viz_cols = ['n_legs', 'mean_abs_edge', 'book_implied_prob', 'builder_hit']
        plot_distributions(
            builder_df,
            columns=builder_viz_cols,
            save_path=plots_dir / "builder_characteristics.png" if args.save_plots else None,
            show=args.show_plots
        )

        logger.info("\n>>> Plotting n_legs distribution...")
        plot_value_counts(
            builder_df,
            'n_legs',
            save_path=plots_dir / "n_legs_distribution.png" if args.save_plots else None,
            show=args.show_plots
        )

        # Add additional features
        if args.add_features:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6: ADDING ADVANCED FEATURES")
            logger.info("=" * 80)

            logger.info("\n>>> Adding stat composition features...")
            builder_df = add_stat_composition_features(leg_df, builder_df)

            logger.info("\n>>> Adding player diversity features...")
            builder_df = add_player_diversity_features(leg_df, builder_df)

            display_dataframe(builder_df, "Builder-Level Dataset - Final", max_rows=15)
            display_summary_stats(builder_df, title="Builder-Level Dataset - Final Statistics")

        # Save processed datasets
        logger.info("\nSaving processed datasets...")

        leg_path = args.output_dir / "bet_builder_legs.parquet"
        builder_path = args.output_dir / "bet_builders.parquet"
        games_path = args.output_dir / "player_games.parquet"

        player_games.to_parquet(games_path, index=False)
        leg_df.to_parquet(leg_path, index=False)
        builder_df.to_parquet(builder_path, index=False)

        logger.info(f" Saved player games to {games_path}")
        logger.info(f" Saved leg-level dataset to {leg_path} ({len(leg_df):,} rows)")
        logger.info(f" Saved builder-level dataset to {builder_path} ({len(builder_df):,} rows)")

        # Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total games: {player_games['game_id'].nunique():,}")
        logger.info(f"Total players: {player_games['player_id'].nunique():,}")
        logger.info(f"Total builders: {builder_df['builder_id'].nunique():,}")
        logger.info(f"Total legs: {len(leg_df):,}")
        logger.info(f"Average legs per builder: {builder_df['n_legs'].mean():.2f}")
        logger.info(f"Builder hit rate: {builder_df['builder_hit'].mean()*100:.2f}%")

        logger.info("\n Data preparation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
