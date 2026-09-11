#!/usr/bin/env python3
"""
Monte Carlo simulation for player points forecasting.

This script runs Monte Carlo simulations to forecast player statistics,
with optional Apple Metal GPU acceleration via MLX for faster execution.

Usage:
    python run_monte_carlo_forecast.py --n-sims 10000 --use-gpu
    python run_monte_carlo_forecast.py --stat pts --player-id 12345
    python run_monte_carlo_forecast.py --compute-all-players --output forecasts.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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
    monte_carlo_config,
)
from src.models.monte_carlo_forecasting import (
    MonteCarloForecaster,
    MonteCarloConfig,
    forecast_all_players_points,
    compare_mc_to_parametric,
    _MLX_AVAILABLE,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for player forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument(
        "--distributions-file",
        type=Path,
        default=MODELS_DIR / "player_stat_distributions.parquet",
        help="Path to fitted distributions file",
    )

    parser.add_argument(
        "--correlations-file",
        type=Path,
        default=MODELS_DIR / "player_stat_correlations.parquet",
        help="Path to player correlations file (optional)",
    )

    parser.add_argument(
        "--prop-lines-file",
        type=Path,
        default=None,
        help="Path to prop lines file for hit probability calculation",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for forecasts",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="mc_forecasts.parquet",
        help="Output filename",
    )

    # Monte Carlo parameters
    parser.add_argument(
        "--n-sims",
        type=int,
        default=monte_carlo_config.n_simulations,
        help="Number of Monte Carlo simulations",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=monte_carlo_config.use_gpu,
        help="Use Apple Metal GPU acceleration (requires MLX)",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU-only execution",
    )

    parser.add_argument(
        "--use-correlations",
        action="store_true",
        default=monte_carlo_config.use_correlations,
        help="Apply correlation structure between stats",
    )

    parser.add_argument(
        "--no-correlations",
        action="store_true",
        help="Disable correlation structure (faster, less accurate)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=monte_carlo_config.random_state,
        help="Random seed for reproducibility",
    )

    # Forecasting options
    parser.add_argument(
        "--stat",
        type=str,
        default="pts",
        choices=["pts", "fg3m", "ast", "reb"],
        help="Stat to forecast",
    )

    parser.add_argument(
        "--player-id",
        type=str,
        default=None,
        help="Specific player ID to forecast (if not set, forecasts all)",
    )

    parser.add_argument(
        "--compute-all-players",
        action="store_true",
        help="Compute forecasts for all players",
    )

    parser.add_argument(
        "--compute-hit-probs",
        action="store_true",
        help="Compute hit probabilities for prop lines",
    )

    parser.add_argument(
        "--compare-parametric",
        action="store_true",
        help="Compare MC probabilities to parametric model",
    )

    # Logging
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


def print_gpu_info() -> None:
    """Print GPU availability information."""
    logger.info("\n" + "=" * 60)
    logger.info("GPU ACCELERATION STATUS")
    logger.info("=" * 60)

    if _MLX_AVAILABLE:
        try:
            import mlx.core as mx
            logger.info("✓ MLX (Apple Metal GPU) is available")
            logger.info(f"  Default device: {mx.default_device()}")
        except Exception as e:
            logger.warning(f"MLX import failed: {e}")
    else:
        logger.info("✗ MLX not available - using CPU fallback")
        logger.info("  To enable GPU: pip install mlx")


def run_single_player_forecast(
    forecaster: MonteCarloForecaster,
    player_id: str,
    stat: str,
    output_dir: Path
) -> pd.DataFrame:
    """Run forecast for a single player."""
    logger.info(f"\nForecasting {stat} for player {player_id}...")

    start_time = time.time()

    # Get full simulation data
    forecast_df = forecaster.forecast_player(player_id, [stat])

    if forecast_df.empty:
        logger.warning(f"No forecast data available for player {player_id}")
        return pd.DataFrame()

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(forecast_df):,} simulations in {elapsed:.2f}s")

    # Compute summary
    summary = forecaster.compute_forecast_summary(player_id, stat)

    logger.info("\nForecast Summary:")
    logger.info(f"  Mean: {summary['mean']:.1f}")
    logger.info(f"  Std Dev: {summary['std']:.1f}")
    logger.info(f"  Median (p50): {summary['p50']:.1f}")
    logger.info(f"  5th percentile: {summary['p5']:.1f}")
    logger.info(f"  95th percentile: {summary['p95']:.1f}")
    logger.info(f"  Range: [{summary['min']:.0f}, {summary['max']:.0f}]")

    # Save simulation data
    out_path = output_dir / f"mc_sims_{player_id}_{stat}.parquet"
    forecast_df.to_parquet(out_path, index=False)
    logger.info(f"\n✓ Saved simulation data to {out_path}")

    return forecast_df


def run_all_players_forecast(
    distributions_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    config: MonteCarloConfig,
    stat: str,
    output_dir: Path
) -> pd.DataFrame:
    """Run forecast for all players."""
    logger.info(f"\nForecasting {stat} for all players...")

    start_time = time.time()

    result_df = forecast_all_players_points(
        distributions_df,
        correlations_df,
        config,
        stat=stat
    )

    elapsed = time.time() - start_time

    if result_df.empty:
        logger.warning("No forecasts generated")
        return pd.DataFrame()

    logger.info(f"Generated forecasts for {len(result_df):,} players in {elapsed:.2f}s")
    logger.info(f"Throughput: {len(result_df) / elapsed:.1f} players/second")

    # Summary statistics
    logger.info("\nAggregate Statistics:")
    logger.info(f"  Average predicted {stat}: {result_df['mean'].mean():.1f}")
    logger.info(f"  Std of predictions: {result_df['mean'].std():.1f}")
    logger.info(f"  Average uncertainty (std): {result_df['std'].mean():.1f}")

    # Top performers
    top_players = result_df.nlargest(10, "mean")
    logger.info(f"\nTop 10 players by predicted {stat}:")
    for _, row in top_players.iterrows():
        logger.info(
            f"  {row['player_id']}: {row['mean']:.1f} ± {row['std']:.1f} "
            f"(p5={row['p5']:.1f}, p95={row['p95']:.1f})"
        )

    return result_df


def run_hit_probability_analysis(
    forecaster: MonteCarloForecaster,
    prop_lines_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Compute Monte Carlo hit probabilities."""
    logger.info("\nComputing hit probabilities for prop lines...")

    start_time = time.time()

    result_df = forecaster.batch_forecast_hit_probabilities(prop_lines_df)

    elapsed = time.time() - start_time

    if result_df.empty:
        logger.warning("No hit probabilities computed")
        return pd.DataFrame()

    logger.info(f"Computed probabilities for {len(result_df):,} props in {elapsed:.2f}s")

    # Summary
    logger.info("\nHit Probability Summary:")
    logger.info(f"  Mean hit probability: {result_df['mc_hit_probability'].mean():.3f}")
    logger.info(f"  Median hit probability: {result_df['mc_hit_probability'].median():.3f}")

    # Group by stat
    by_stat = result_df.groupby("stat")["mc_hit_probability"].agg(["mean", "std", "count"])
    logger.info("\nBy stat type:")
    for stat, row in by_stat.iterrows():
        logger.info(f"  {stat}: mean={row['mean']:.3f}, std={row['std']:.3f}, n={row['count']:.0f}")

    return result_df


def main() -> int:
    """Main execution function."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(__import__("logging"), log_level), log_file=args.log_file)

    logger.info("=" * 80)
    logger.info("MONTE CARLO PLAYER FORECASTING")
    logger.info("=" * 80)

    # Print GPU info
    print_gpu_info()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Build configuration
        config = MonteCarloConfig(
            n_simulations=args.n_sims,
            random_state=args.random_state,
            use_gpu=args.use_gpu and not args.no_gpu,
            use_correlations=args.use_correlations and not args.no_correlations,
        )

        logger.info("\nConfiguration:")
        logger.info(f"  Simulations: {config.n_simulations:,}")
        logger.info(f"  Use GPU: {config.use_gpu}")
        logger.info(f"  Use correlations: {config.use_correlations}")
        logger.info(f"  Random state: {config.random_state}")

        # Load distributions
        logger.info(f"\nLoading distributions from {args.distributions_file}...")
        if not args.distributions_file.exists():
            raise FileNotFoundError(
                f"Distributions file not found: {args.distributions_file}\n"
                "Run 'python scripts/fit_player_distributions.py' first."
            )

        distributions_df = pd.read_parquet(args.distributions_file)
        logger.info(f"Loaded {len(distributions_df):,} player-stat distributions")

        # Load correlations (optional)
        correlations_df = None
        if config.use_correlations and args.correlations_file.exists():
            logger.info(f"Loading correlations from {args.correlations_file}...")
            correlations_df = pd.read_parquet(args.correlations_file)
            logger.info(f"Loaded {len(correlations_df):,} correlation records")
        elif config.use_correlations:
            logger.warning(
                f"Correlations file not found: {args.correlations_file}\n"
                "Proceeding without correlation structure."
            )

        # Create forecaster
        forecaster = MonteCarloForecaster(distributions_df, correlations_df, config)

        logger.info(f"\nBackend: {forecaster.backend.upper()}")

        # Run appropriate analysis
        result_df = pd.DataFrame()

        if args.player_id:
            # Single player forecast
            result_df = run_single_player_forecast(
                forecaster, args.player_id, args.stat, args.output_dir
            )

        elif args.compute_all_players:
            # All players forecast
            result_df = run_all_players_forecast(
                distributions_df,
                correlations_df,
                config,
                args.stat,
                args.output_dir
            )

            if not result_df.empty:
                out_path = args.output_dir / args.output_file
                result_df.to_parquet(out_path, index=False)
                logger.info(f"\n✓ Saved forecasts to {out_path}")

        elif args.compute_hit_probs and args.prop_lines_file:
            # Hit probability analysis
            if not args.prop_lines_file.exists():
                raise FileNotFoundError(f"Prop lines file not found: {args.prop_lines_file}")

            prop_lines_df = pd.read_parquet(args.prop_lines_file)
            result_df = run_hit_probability_analysis(forecaster, prop_lines_df, args.output_dir)

            if not result_df.empty:
                out_path = args.output_dir / "mc_hit_probabilities.parquet"
                result_df.to_parquet(out_path, index=False)
                logger.info(f"\n✓ Saved hit probabilities to {out_path}")

        elif args.compare_parametric and args.prop_lines_file:
            # Compare MC to parametric
            if not args.prop_lines_file.exists():
                raise FileNotFoundError(f"Prop lines file not found: {args.prop_lines_file}")

            prop_lines_df = pd.read_parquet(args.prop_lines_file)
            result_df = compare_mc_to_parametric(
                distributions_df,
                prop_lines_df,
                correlations_df,
                config
            )

            if not result_df.empty:
                out_path = args.output_dir / "mc_vs_parametric.parquet"
                result_df.to_parquet(out_path, index=False)
                logger.info(f"\n✓ Saved comparison to {out_path}")

        else:
            # Default: quick demo with sample player
            logger.info("\nRunning quick demo (use --compute-all-players for full analysis)...")

            # Get a sample player
            sample_players = distributions_df[
                distributions_df["stat"] == args.stat
            ]["player_id"].head(5).tolist()

            if sample_players:
                logger.info(f"\nSample forecasts for {len(sample_players)} players:")
                for player_id in sample_players:
                    summary = forecaster.compute_forecast_summary(str(player_id), args.stat)
                    if summary:
                        logger.info(
                            f"  {player_id}: {summary['mean']:.1f} ± {summary['std']:.1f} "
                            f"(range: {summary['min']:.0f}-{summary['max']:.0f})"
                        )

        logger.info("\n" + "=" * 80)
        logger.info("✓ Monte Carlo forecasting completed successfully!")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error during Monte Carlo forecasting: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
