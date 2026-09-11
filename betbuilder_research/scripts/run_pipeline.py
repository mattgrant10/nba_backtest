#!/usr/bin/env python3
"""
Unified Pipeline Execution Script for NBA Bet Builder Analysis.

This script provides two execution modes:

1. FULL GPU MODE (--mode full):
   - Runs the entire workflow end-to-end
   - Includes data preparation, distribution fitting, correlation analysis,
     Monte Carlo simulation (GPU-accelerated), model training, and backtesting

2. SELECTIVE EXECUTION MODE (--mode selective):
   - Allows running only specific sections via --steps flag
   - Available steps:
     a) 'data'        - Data preparation only
     b) 'distributions' - Distribution fitting only
     c) 'correlations' - Correlation analysis only
     d) 'monte-carlo' - GPU-based Monte Carlo simulation only
     e) 'train'       - Model training only
     f) 'backtest'    - Backtesting only
     g) 'analysis'    - All non-simulation analysis (data + distributions + correlations)
     h) 'simulation'  - Monte Carlo simulation only (alias for monte-carlo)

Usage Examples:
    # Full GPU mode (everything)
    python run_pipeline.py --mode full

    # Selective: Only Monte Carlo simulation
    python run_pipeline.py --mode selective --steps monte-carlo

    # Selective: Only analysis (no simulation)
    python run_pipeline.py --mode selective --steps analysis

    # Selective: Multiple specific steps
    python run_pipeline.py --mode selective --steps data distributions monte-carlo

    # Full mode with custom simulation count
    python run_pipeline.py --mode full --n-sims 50000

    # Disable GPU (CPU fallback)
    python run_pipeline.py --mode full --no-gpu
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    setup_logging,
    get_logger,
    model_config,
    backtest_config,
    monte_carlo_config,
)


logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Pipeline execution modes."""
    FULL = "full"
    SELECTIVE = "selective"


class PipelineStep(Enum):
    """Individual pipeline steps."""
    DATA = "data"
    DISTRIBUTIONS = "distributions"
    CORRELATIONS = "correlations"
    MONTE_CARLO = "monte-carlo"
    EDGES = "edges"  # Calculate break-even odds for each prop
    TRAIN = "train"
    BACKTEST = "backtest"

    # Composite steps
    ANALYSIS = "analysis"  # data + distributions + correlations
    SIMULATION = "simulation"  # alias for monte-carlo


# Step dependencies - which steps must run before others
STEP_DEPENDENCIES: Dict[PipelineStep, List[PipelineStep]] = {
    PipelineStep.DATA: [],
    PipelineStep.DISTRIBUTIONS: [PipelineStep.DATA],
    PipelineStep.CORRELATIONS: [PipelineStep.DATA],
    PipelineStep.MONTE_CARLO: [PipelineStep.DISTRIBUTIONS, PipelineStep.CORRELATIONS],
    PipelineStep.EDGES: [PipelineStep.DISTRIBUTIONS],
    PipelineStep.TRAIN: [PipelineStep.DATA],
    PipelineStep.BACKTEST: [PipelineStep.TRAIN],
}

# Composite step expansions
COMPOSITE_STEPS: Dict[PipelineStep, List[PipelineStep]] = {
    PipelineStep.ANALYSIS: [
        PipelineStep.DATA,
        PipelineStep.DISTRIBUTIONS,
        PipelineStep.CORRELATIONS,
    ],
    PipelineStep.SIMULATION: [PipelineStep.MONTE_CARLO],
}

# Steps that use GPU
GPU_STEPS: Set[PipelineStep] = {PipelineStep.MONTE_CARLO}


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    # Execution mode
    mode: ExecutionMode = ExecutionMode.FULL
    steps: List[PipelineStep] = field(default_factory=list)

    # GPU settings
    use_gpu: bool = True

    # Monte Carlo settings
    n_simulations: int = 10000
    use_correlations: bool = True
    forecast_stat: str = "pts"

    # Data settings
    min_games: int = 20
    validate_data: bool = True
    save_plots: bool = False
    show_plots: bool = False

    # Backtest settings
    train_window_days: int = 60
    test_window_days: int = 7
    save_predictions: bool = False

    # General
    random_state: int = 42
    verbose: bool = False
    output_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR

    def get_effective_steps(self) -> List[PipelineStep]:
        """Get the effective list of steps to run, expanding composites."""
        if self.mode == ExecutionMode.FULL:
            return [
                PipelineStep.DATA,
                PipelineStep.DISTRIBUTIONS,
                PipelineStep.CORRELATIONS,
                PipelineStep.MONTE_CARLO,
                PipelineStep.EDGES,
                PipelineStep.TRAIN,
                PipelineStep.BACKTEST,
            ]

        # Expand composite steps
        effective = []
        for step in self.steps:
            if step in COMPOSITE_STEPS:
                effective.extend(COMPOSITE_STEPS[step])
            else:
                effective.append(step)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for step in effective:
            if step not in seen:
                seen.add(step)
                unique.append(step)

        return unique


class PipelineRunner:
    """Orchestrates pipeline execution."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results: Dict[str, dict] = {}
        self.timings: Dict[str, float] = {}

    def run(self) -> int:
        """Run the pipeline according to configuration."""
        steps = self.config.get_effective_steps()

        logger.info("=" * 80)
        logger.info("NBA BET BUILDER PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.config.mode.value.upper()}")
        logger.info(f"Steps to run: {[s.value for s in steps]}")
        logger.info(f"GPU enabled: {self.config.use_gpu}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 80)

        total_start = time.time()

        try:
            for step in steps:
                self._run_step(step)

            total_elapsed = time.time() - total_start

            self._print_summary(total_elapsed)

            return 0

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return 1

    def _run_step(self, step: PipelineStep) -> None:
        """Run a single pipeline step."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STEP: {step.value.upper()}")
        logger.info("=" * 80)

        start_time = time.time()

        step_handlers = {
            PipelineStep.DATA: self._run_data_preparation,
            PipelineStep.DISTRIBUTIONS: self._run_distribution_fitting,
            PipelineStep.CORRELATIONS: self._run_correlation_analysis,
            PipelineStep.MONTE_CARLO: self._run_monte_carlo,
            PipelineStep.EDGES: self._run_edge_calculation,
            PipelineStep.TRAIN: self._run_model_training,
            PipelineStep.BACKTEST: self._run_backtest,
        }

        handler = step_handlers.get(step)
        if handler:
            result = handler()
            self.results[step.value] = result
        else:
            logger.warning(f"No handler for step: {step.value}")

        elapsed = time.time() - start_time
        self.timings[step.value] = elapsed

        logger.info(f"Step '{step.value}' completed in {elapsed:.2f}s")

    def _run_data_preparation(self) -> dict:
        """Run data preparation step."""
        import pandas as pd
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

        logger.info("Loading raw data...")
        player_games = load_player_game_logs()
        prop_lines = load_prop_lines(player_games=player_games)
        bet_builder_legs = load_bet_builder_legs(
            prop_lines=prop_lines, player_games=player_games
        )

        if self.config.validate_data:
            logger.info("Validating data consistency...")
            validation = validate_data_consistency(player_games, prop_lines, bet_builder_legs)
            for warning in validation.get("warnings", []):
                logger.warning(warning)

        logger.info("Adding context features...")
        player_games = add_basic_context_features(player_games)

        logger.info("Building leg-level dataset...")
        leg_df = build_leg_level_dataset(player_games, prop_lines, bet_builder_legs)

        logger.info("Building builder-level dataset...")
        builder_df = build_builder_level_dataset(leg_df)

        logger.info("Adding advanced features...")
        builder_df = add_stat_composition_features(leg_df, builder_df)
        builder_df = add_player_diversity_features(leg_df, builder_df)

        # Save processed data
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        games_path = self.config.output_dir / "player_games.parquet"
        leg_path = self.config.output_dir / "bet_builder_legs.parquet"
        builder_path = self.config.output_dir / "bet_builders.parquet"

        player_games.to_parquet(games_path, index=False)
        leg_df.to_parquet(leg_path, index=False)
        builder_df.to_parquet(builder_path, index=False)

        logger.info(f"Saved player games: {games_path}")
        logger.info(f"Saved legs: {leg_path} ({len(leg_df):,} rows)")
        logger.info(f"Saved builders: {builder_path} ({len(builder_df):,} rows)")

        return {
            "n_games": len(player_games),
            "n_players": player_games["player_id"].nunique(),
            "n_legs": len(leg_df),
            "n_builders": len(builder_df),
        }

    def _run_distribution_fitting(self) -> dict:
        """Run distribution fitting step."""
        import pandas as pd
        from src.models.stat_distributions import (
            summarise_player_stat_distributions,
            attach_parametric_models,
        )

        games_path = self.config.output_dir / "player_games.parquet"
        if not games_path.exists():
            raise FileNotFoundError(
                f"Player games not found: {games_path}\n"
                "Run the 'data' step first."
            )

        player_games = pd.read_parquet(games_path)

        logger.info(f"Computing distributions (min_games={self.config.min_games})...")
        summary = summarise_player_stat_distributions(
            player_games,
            stat_cols=model_config.stat_cols,
            min_games=self.config.min_games,
        )

        logger.info("Fitting parametric models...")
        summary_with_models = attach_parametric_models(summary)

        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.models_dir / "player_stat_distributions.parquet"
        summary_with_models.to_parquet(out_path, index=False)

        logger.info(f"Saved distributions: {out_path}")

        family_counts = summary_with_models["family"].value_counts()
        logger.info("Model family distribution:")
        for family, count in family_counts.items():
            pct = count / len(summary_with_models) * 100
            logger.info(f"  {family}: {count} ({pct:.1f}%)")

        return {
            "n_distributions": len(summary_with_models),
            "n_players": summary_with_models["player_id"].nunique(),
            "families": family_counts.to_dict(),
        }

    def _run_correlation_analysis(self) -> dict:
        """Run correlation analysis step."""
        import pandas as pd
        from src.models.dependencies import (
            compute_player_stat_correlations,
            compute_global_correlation,
            identify_high_correlation_pairs,
        )

        games_path = self.config.output_dir / "player_games.parquet"
        if not games_path.exists():
            raise FileNotFoundError(
                f"Player games not found: {games_path}\n"
                "Run the 'data' step first."
            )

        player_games = pd.read_parquet(games_path)

        logger.info("Computing per-player correlations...")
        per_player_corr = compute_player_stat_correlations(
            player_games,
            stat_cols=model_config.stat_cols,
            min_games=self.config.min_games,
            method=model_config.correlation_method,
        )

        logger.info("Computing global correlations...")
        global_corr = compute_global_correlation(
            player_games,
            stat_cols=model_config.stat_cols,
            method=model_config.correlation_method,
        )

        logger.info("Identifying high correlation pairs...")
        high_corr = identify_high_correlation_pairs(per_player_corr, threshold=0.7)

        # Save results
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        per_player_path = self.config.models_dir / "player_stat_correlations.parquet"
        global_path = self.config.models_dir / "global_stat_correlations.parquet"

        per_player_corr.to_parquet(per_player_path, index=False)
        global_corr.to_parquet(global_path, index=False)

        logger.info(f"Saved per-player correlations: {per_player_path}")
        logger.info(f"Saved global correlations: {global_path}")

        if len(high_corr) > 0:
            high_corr_path = self.config.models_dir / "high_correlations.parquet"
            high_corr.to_parquet(high_corr_path, index=False)
            logger.info(f"Saved high correlations: {high_corr_path}")

        return {
            "n_player_correlations": len(per_player_corr),
            "n_players_analyzed": per_player_corr["player_id"].nunique(),
            "n_high_correlations": len(high_corr),
        }

    def _run_monte_carlo(self) -> dict:
        """Run Monte Carlo simulation step (GPU-accelerated)."""
        import pandas as pd
        from src.models.monte_carlo_forecasting import (
            MonteCarloForecaster,
            MonteCarloConfig,
            forecast_all_players_points,
            _MLX_AVAILABLE,
        )

        # Check GPU availability
        if self.config.use_gpu and not _MLX_AVAILABLE:
            logger.warning("MLX not available, falling back to CPU")
            logger.warning("Install MLX for GPU acceleration: pip install mlx")

        backend = "GPU (Metal)" if self.config.use_gpu and _MLX_AVAILABLE else "CPU"
        logger.info(f"Backend: {backend}")
        logger.info(f"Simulations: {self.config.n_simulations:,}")
        logger.info(f"Use correlations: {self.config.use_correlations}")

        # Load required data
        dist_path = self.config.models_dir / "player_stat_distributions.parquet"
        corr_path = self.config.models_dir / "player_stat_correlations.parquet"

        if not dist_path.exists():
            raise FileNotFoundError(
                f"Distributions not found: {dist_path}\n"
                "Run the 'distributions' step first."
            )

        distributions_df = pd.read_parquet(dist_path)

        correlations_df = None
        if self.config.use_correlations and corr_path.exists():
            correlations_df = pd.read_parquet(corr_path)
            logger.info(f"Loaded {len(correlations_df):,} correlation records")

        # Configure Monte Carlo
        mc_config = MonteCarloConfig(
            n_simulations=self.config.n_simulations,
            random_state=self.config.random_state,
            use_gpu=self.config.use_gpu,
            use_correlations=self.config.use_correlations,
        )

        # Run forecasting
        logger.info(f"\nForecasting {self.config.forecast_stat} for all players...")

        result_df = forecast_all_players_points(
            distributions_df,
            correlations_df,
            mc_config,
            stat=self.config.forecast_stat,
        )

        if result_df.empty:
            logger.warning("No forecasts generated")
            return {"n_forecasts": 0}

        # Save results
        out_path = self.config.models_dir / "mc_forecasts.parquet"
        result_df.to_parquet(out_path, index=False)
        logger.info(f"Saved forecasts: {out_path}")

        # Summary statistics
        logger.info(f"\nForecast Summary ({self.config.forecast_stat}):")
        logger.info(f"  Players forecasted: {len(result_df):,}")
        logger.info(f"  Average prediction: {result_df['mean'].mean():.1f}")
        logger.info(f"  Average uncertainty: {result_df['std'].mean():.1f}")

        # Top performers
        top_5 = result_df.nlargest(5, "mean")
        logger.info(f"\nTop 5 predicted {self.config.forecast_stat}:")
        for _, row in top_5.iterrows():
            logger.info(
                f"  {row['player_id']}: {row['mean']:.1f} ± {row['std']:.1f}"
            )

        return {
            "n_forecasts": len(result_df),
            "backend": backend,
            "n_simulations": self.config.n_simulations,
            "avg_prediction": float(result_df["mean"].mean()),
            "avg_uncertainty": float(result_df["std"].mean()),
        }

    def _run_edge_calculation(self) -> dict:
        """Run edge calculation step - calculate break-even odds for each prop."""
        import pandas as pd
        from src.models.edge_calculator import (
            PlayerEdgeCalculator,
            calculate_breakeven_odds,
        )

        logger.info("Calculating break-even odds for player props...")

        # Load distributions
        dist_path = self.config.models_dir / "player_stat_distributions.parquet"
        games_path = self.config.output_dir / "player_games.parquet"

        if not dist_path.exists():
            raise FileNotFoundError(
                f"Distributions not found: {dist_path}\n"
                "Run the 'distributions' step first."
            )

        distributions_df = pd.read_parquet(dist_path)

        # Load player games for names
        player_games_df = None
        if games_path.exists():
            player_games_df = pd.read_parquet(games_path)

        # Calculate edges
        edge_path = self.config.models_dir / "player_edges.parquet"
        edge_df = calculate_breakeven_odds(
            distributions_df,
            player_games_df,
            stats=model_config.stat_cols,
            output_path=str(edge_path),
        )

        if len(edge_df) == 0:
            logger.warning("No edges calculated")
            return {"n_edges": 0}

        # Summary
        logger.info(f"Calculated edges for {len(edge_df):,} props")

        # Show best value props (highest probability = lowest min odds)
        logger.info("\nTop Value Props (highest probability of hitting):")

        for stat in model_config.stat_cols:
            stat_df = edge_df[edge_df["stat"] == stat]
            if len(stat_df) == 0:
                continue

            over_best = stat_df[stat_df["direction"] == "over"].nlargest(3, "probability")
            under_best = stat_df[stat_df["direction"] == "under"].nlargest(3, "probability")

            logger.info(f"\n  {stat.upper()}:")
            for _, row in over_best.iterrows():
                name = row.get("player_name") or row["player_id"]
                logger.info(
                    f"    Over {row['line']}: {name} "
                    f"({row['probability']:.1%}, need {row['min_odds_for_edge']:.2f}+ odds)"
                )
            for _, row in under_best.iterrows():
                name = row.get("player_name") or row["player_id"]
                logger.info(
                    f"    Under {row['line']}: {name} "
                    f"({row['probability']:.1%}, need {row['min_odds_for_edge']:.2f}+ odds)"
                )

        return {
            "n_edges": len(edge_df),
            "n_players": edge_df["player_id"].nunique(),
            "n_stats": edge_df["stat"].nunique(),
            "avg_probability": float(edge_df["probability"].mean()),
            "avg_min_odds": float(edge_df["min_odds_for_edge"].mean()),
        }

    def _run_model_training(self) -> dict:
        """Run model training step."""
        import pandas as pd
        import joblib
        from src.models.bet_builder_outcome import (
            train_builder_hit_model,
            default_feature_columns,
        )

        builder_path = self.config.output_dir / "bet_builders.parquet"
        if not builder_path.exists():
            raise FileNotFoundError(
                f"Builders not found: {builder_path}\n"
                "Run the 'data' step first."
            )

        builders = pd.read_parquet(builder_path)
        feature_cols = default_feature_columns()

        logger.info(f"Training on {len(builders):,} builders")
        logger.info(f"Features: {feature_cols}")

        model, metrics = train_builder_hit_model(
            builders,
            feature_cols=feature_cols,
            label_col="builder_hit",
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        # Save model
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.models_dir / "builder_hit_model.joblib"
        joblib.dump(model, model_path)

        logger.info(f"Saved model: {model_path}")

        logger.info(f"\nTraining Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier']:.4f}")

        return {
            "n_samples": metrics["n_obs"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "brier": metrics["brier"],
        }

    def _run_backtest(self) -> dict:
        """Run backtesting step."""
        import pandas as pd
        from src.backtest.simulator import run_rolling_backtest, analyze_backtest_results
        from src.backtest.splits import TimeSplitConfig
        from src.models.bet_builder_outcome import default_feature_columns

        builder_path = self.config.output_dir / "bet_builders.parquet"
        if not builder_path.exists():
            raise FileNotFoundError(
                f"Builders not found: {builder_path}\n"
                "Run the 'data' step first."
            )

        builders = pd.read_parquet(builder_path)

        split_cfg = TimeSplitConfig(
            date_col=backtest_config.date_col,
            train_window_days=self.config.train_window_days,
            test_window_days=self.config.test_window_days,
            min_train_size=backtest_config.min_train_size,
        )

        logger.info(f"Train window: {split_cfg.train_window_days} days")
        logger.info(f"Test window: {split_cfg.test_window_days} days")

        feature_cols = default_feature_columns()

        results_df, predictions_df = run_rolling_backtest(
            builders,
            split_cfg=split_cfg,
            feature_cols=feature_cols,
            label_col="builder_hit",
            random_state=self.config.random_state,
            save_predictions=self.config.save_predictions,
        )

        # Save results
        results_path = self.config.output_dir / "backtest_results.parquet"
        results_df.to_parquet(results_path, index=False)
        logger.info(f"Saved backtest results: {results_path}")

        if self.config.save_predictions and predictions_df is not None:
            pred_path = self.config.output_dir / "backtest_predictions.parquet"
            predictions_df.to_parquet(pred_path, index=False)
            logger.info(f"Saved predictions: {pred_path}")

        if len(results_df) > 0:
            overall_roi = results_df["profit"].sum() / results_df["total_stake"].sum()
            profitable_folds = (results_df["roi"] > 0).sum()

            logger.info(f"\nBacktest Results:")
            logger.info(f"  Folds: {len(results_df)}")
            logger.info(f"  Overall ROI: {overall_roi*100:.2f}%")
            logger.info(f"  Profitable folds: {profitable_folds}/{len(results_df)}")
            logger.info(f"  Total profit: {results_df['profit'].sum():.2f}")

            return {
                "n_folds": len(results_df),
                "overall_roi": overall_roi,
                "total_profit": float(results_df["profit"].sum()),
                "profitable_folds": profitable_folds,
            }

        return {"n_folds": 0}

    def _print_summary(self, total_elapsed: float) -> None:
        """Print execution summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        logger.info(f"\nTotal execution time: {total_elapsed:.2f}s")

        logger.info("\nStep timings:")
        for step, elapsed in self.timings.items():
            pct = elapsed / total_elapsed * 100
            logger.info(f"  {step}: {elapsed:.2f}s ({pct:.1f}%)")

        logger.info("\nStep results:")
        for step, result in self.results.items():
            logger.info(f"  {step}:")
            for key, value in result.items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.4f}")
                else:
                    logger.info(f"    {key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NBA Bet Builder Pipeline - Full or Selective Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full GPU mode (everything)
  python run_pipeline.py --mode full

  # Selective: Only Monte Carlo simulation
  python run_pipeline.py --mode selective --steps monte-carlo

  # Selective: Only analysis (no simulation)
  python run_pipeline.py --mode selective --steps analysis

  # Selective: Multiple specific steps
  python run_pipeline.py --mode selective --steps data distributions monte-carlo

  # Full mode with more simulations
  python run_pipeline.py --mode full --n-sims 50000

  # Disable GPU
  python run_pipeline.py --mode full --no-gpu

Step aliases:
  analysis   = data + distributions + correlations
  simulation = monte-carlo
        """
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "selective"],
        default="full",
        help="Execution mode: 'full' runs everything, 'selective' runs specified steps",
    )

    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=[
            "data", "distributions", "correlations", "monte-carlo", "edges",
            "train", "backtest", "analysis", "simulation"
        ],
        default=None,
        help="Steps to run in selective mode",
    )

    # GPU settings
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration for Monte Carlo (default: True)",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU, use CPU only",
    )

    # Monte Carlo settings
    parser.add_argument(
        "--n-sims",
        type=int,
        default=monte_carlo_config.n_simulations,
        help="Number of Monte Carlo simulations",
    )

    parser.add_argument(
        "--no-correlations",
        action="store_true",
        help="Disable correlation structure in Monte Carlo",
    )

    parser.add_argument(
        "--stat",
        type=str,
        choices=["pts", "fg3m", "ast", "reb"],
        default="pts",
        help="Stat to forecast in Monte Carlo",
    )

    # Data settings
    parser.add_argument(
        "--min-games",
        type=int,
        default=model_config.min_games,
        help="Minimum games required per player",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively",
    )

    # Backtest settings
    parser.add_argument(
        "--train-window",
        type=int,
        default=backtest_config.train_window_days,
        help="Training window in days",
    )

    parser.add_argument(
        "--test-window",
        type=int,
        default=backtest_config.test_window_days,
        help="Test window in days",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions from backtest",
    )

    # General settings
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for models",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(
        level=getattr(__import__("logging"), log_level),
        log_file=args.log_file
    )

    # Validate arguments
    if args.mode == "selective" and not args.steps:
        logger.error("Selective mode requires --steps argument")
        logger.error("Example: --mode selective --steps monte-carlo")
        return 1

    # Build configuration
    steps = []
    if args.steps:
        for step_name in args.steps:
            try:
                steps.append(PipelineStep(step_name))
            except ValueError:
                logger.error(f"Invalid step: {step_name}")
                return 1

    config = PipelineConfig(
        mode=ExecutionMode(args.mode),
        steps=steps,
        use_gpu=args.use_gpu and not args.no_gpu,
        n_simulations=args.n_sims,
        use_correlations=not args.no_correlations,
        forecast_stat=args.stat,
        min_games=args.min_games,
        validate_data=not args.no_validate,
        save_plots=args.save_plots,
        show_plots=args.show_plots,
        train_window_days=args.train_window,
        test_window_days=args.test_window,
        save_predictions=args.save_predictions,
        random_state=args.random_state,
        verbose=args.verbose,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )

    # Run pipeline
    runner = PipelineRunner(config)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
