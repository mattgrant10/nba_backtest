#!/usr/bin/env python3
"""
Calculate Break-Even Odds for Player Props.

This script calculates the minimum decimal odds required for positive expected
value on each player prop, based on their fitted statistical distributions.

Output format:
    Player: LeBron James
    Line: Over 24.5 pts
    True Probability: 52%
    Min Odds for +EV: 1.92

Interpretation: Need odds >= 1.92 on LeBron over 24.5 pts for a positive edge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    setup_logging,
    get_logger,
    model_config,
)
from src.models.edge_calculator import (
    PlayerEdgeCalculator,
    calculate_breakeven_odds,
    format_edge_summary,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate break-even odds for player props",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate edges for all players
  python calculate_edges.py

  # Calculate for specific player
  python calculate_edges.py --player-id 203507

  # Filter by stat
  python calculate_edges.py --stats pts ast

  # Show top value props (lowest min odds = highest probability)
  python calculate_edges.py --top 50

  # Output as CSV
  python calculate_edges.py --output-format csv --output edges.csv
        """,
    )

    parser.add_argument(
        "--distributions-file",
        type=Path,
        default=MODELS_DIR / "player_stat_distributions.parquet",
        help="Path to fitted distributions file",
    )

    parser.add_argument(
        "--games-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "player_games.parquet",
        help="Path to player games file (for names)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "player_edges.parquet",
        help="Output file path",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Output format",
    )

    parser.add_argument(
        "--player-id",
        type=str,
        default=None,
        help="Calculate edges for specific player only",
    )

    parser.add_argument(
        "--stats",
        type=str,
        nargs="+",
        default=None,
        help="Stats to analyze (default: pts fg3m ast reb)",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Show top N edges in summary",
    )

    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.3,
        help="Filter props with probability below this (default: 0.3)",
    )

    parser.add_argument(
        "--max-odds",
        type=float,
        default=5.0,
        help="Filter props requiring odds above this (default: 5.0)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(__import__("logging"), log_level))

    logger.info("=" * 80)
    logger.info("PLAYER PROP EDGE CALCULATOR")
    logger.info("=" * 80)

    try:
        # Load distributions
        if not args.distributions_file.exists():
            logger.error(f"Distributions file not found: {args.distributions_file}")
            logger.error("Run the pipeline first: python scripts/run_pipeline.py --mode full")
            return 1

        logger.info(f"Loading distributions from {args.distributions_file}...")
        distributions_df = pd.read_parquet(args.distributions_file)
        logger.info(f"Loaded {len(distributions_df):,} player-stat distributions")

        # Load player games for names
        player_games_df = None
        if args.games_file.exists():
            player_games_df = pd.read_parquet(args.games_file)
            logger.info(f"Loaded player names from {args.games_file}")

        # Filter by player if specified
        if args.player_id:
            distributions_df = distributions_df[
                distributions_df["player_id"].astype(str) == str(args.player_id)
            ]
            if len(distributions_df) == 0:
                logger.error(f"Player {args.player_id} not found in distributions")
                return 1
            logger.info(f"Analyzing player {args.player_id} only")

        # Stats to analyze
        stats = args.stats or model_config.stat_cols
        logger.info(f"Analyzing stats: {stats}")

        # Build player name lookup
        player_names = {}
        if player_games_df is not None and "player_name" in player_games_df.columns:
            name_df = player_games_df[["player_id", "player_name"]].drop_duplicates()
            player_names = dict(zip(
                name_df["player_id"].astype(str),
                name_df["player_name"]
            ))

        # Initialize calculator
        calculator = PlayerEdgeCalculator(distributions_df, player_names)

        # Generate report
        logger.info("\nCalculating break-even odds for all props...")

        if args.player_id:
            edge_df = calculator.generate_player_report(
                args.player_id, stats=stats
            )
        else:
            edge_df = calculator.generate_all_players_report(
                stats=stats,
                min_probability=args.min_probability,
                max_min_odds=args.max_odds,
            )

        if len(edge_df) == 0:
            logger.warning("No edges calculated")
            return 0

        logger.info(f"Calculated {len(edge_df):,} prop edges")

        # Save output
        args.output.parent.mkdir(parents=True, exist_ok=True)

        if args.output_format in ["parquet", "both"]:
            edge_df.to_parquet(args.output, index=False)
            logger.info(f"Saved edges to {args.output}")

        if args.output_format in ["csv", "both"]:
            csv_path = args.output.with_suffix(".csv")
            edge_df.to_csv(csv_path, index=False)
            logger.info(f"Saved edges to {csv_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EDGE SUMMARY")
        logger.info("=" * 80)

        # Best over props (highest probability)
        logger.info("\nTop OVER Props (highest probability = lowest min odds):")
        over_df = edge_df[edge_df["direction"] == "over"].nlargest(args.top, "probability")

        for _, row in over_df.head(15).iterrows():
            name = row.get("player_name") or row["player_id"]
            logger.info(
                f"  {name}: Over {row['line']} {row['stat']} | "
                f"Prob: {row['probability']:.1%} | "
                f"Need {row['min_odds_for_edge']:.2f}+ odds for +EV"
            )

        # Best under props
        logger.info("\nTop UNDER Props (highest probability = lowest min odds):")
        under_df = edge_df[edge_df["direction"] == "under"].nlargest(args.top, "probability")

        for _, row in under_df.head(15).iterrows():
            name = row.get("player_name") or row["player_id"]
            logger.info(
                f"  {name}: Under {row['line']} {row['stat']} | "
                f"Prob: {row['probability']:.1%} | "
                f"Need {row['min_odds_for_edge']:.2f}+ odds for +EV"
            )

        # Stats summary
        logger.info("\n" + "-" * 80)
        logger.info("STATS BY CATEGORY:")

        for stat in stats:
            stat_df = edge_df[edge_df["stat"] == stat]
            if len(stat_df) == 0:
                continue

            avg_prob = stat_df["probability"].mean()
            avg_odds = stat_df[stat_df["min_odds_for_edge"] < 10]["min_odds_for_edge"].mean()

            logger.info(
                f"  {stat.upper()}: {len(stat_df):,} props | "
                f"Avg prob: {avg_prob:.1%} | Avg min odds: {avg_odds:.2f}"
            )

        logger.info("\n" + "=" * 80)
        logger.info("HOW TO USE THIS DATA:")
        logger.info("=" * 80)
        logger.info("""
  1. Find a player prop you want to bet on
  2. Look up their min_odds_for_edge for that prop
  3. Compare to the odds being offered by your sportsbook
  4. If offered odds >= min_odds_for_edge, you have a +EV bet

  Example:
    - Trae Young Over 24.5 pts has min_odds_for_edge = 1.85
    - Sportsbook offers 1.95 on Over 24.5 pts
    - 1.95 >= 1.85, so this is a +EV bet

    Edge = (1/1.85 - 1/1.95) / (1/1.95) = +5.4%
        """)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
