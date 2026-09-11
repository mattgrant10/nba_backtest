#!/usr/bin/env python3
"""
Enhanced Pipeline Execution Script with Research-Grade Reporting.

Console-first output with:
- Structured tables (≤5 columns, HEAD/TAIL/FOOTER)
- Pop-up visualizations
- Live Monte Carlo feedback
- Data health diagnostics
- Optional HTML report export

Usage:
    python run_pipeline_enhanced.py --mode full --verbosity full
    python run_pipeline_enhanced.py --mode selective --steps distributions edges
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    FIGURES_DIR,
    setup_logging,
    get_logger,
    model_config,
    backtest_config,
    monte_carlo_config,
)

from src.reporting import (
    TableFormatter,
    print_table,
    print_step_header,
    print_step_footer,
    print_warning_block,
    DataDiagnostics,
    run_data_health_checks,
    VisualManager,
    ReportExporter,
    RunMetadata,
    prompt_save_report,
    PDFReportExporter,
    PDFReportConfig,
    ConsoleCapture,
    create_tiered_tables,
    get_tier_description,
    DATA_FEED_DOCUMENTATION,
)

logger = get_logger(__name__)


class Verbosity(Enum):
    LITE = "lite"
    FULL = "full"


class ExecutionMode(Enum):
    FULL = "full"
    SELECTIVE = "selective"


class PipelineStep(Enum):
    DATA = "data"
    DISTRIBUTIONS = "distributions"
    CORRELATIONS = "correlations"
    MONTE_CARLO = "monte-carlo"
    EDGES = "edges"
    TRAIN = "train"
    BACKTEST = "backtest"
    ANALYSIS = "analysis"
    SIMULATION = "simulation"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    verbosity: Verbosity = Verbosity.FULL
    steps: List[PipelineStep] = field(default_factory=list)
    use_gpu: bool = True
    n_simulations: int = 10000
    use_correlations: bool = True
    forecast_stat: str = "pts"
    min_games: int = 20
    validate_data: bool = True
    save_plots: bool = True
    show_plots: bool = True
    train_window_days: int = 60
    test_window_days: int = 7
    save_predictions: bool = False
    random_state: int = 42
    verbose: bool = False
    output_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    figures_dir: Path = FIGURES_DIR

    def get_effective_steps(self) -> List[PipelineStep]:
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
        return self.steps


class EnhancedPipelineRunner:
    """
    Enhanced pipeline runner with research-grade reporting.

    Implements the console-first reporting specification with:
    - 5-column table limit
    - HEAD/TAIL/FOOTER for all tables
    - Pop-up visualizations
    - Data health diagnostics
    - Live Monte Carlo feedback
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results: Dict[str, dict] = {}
        self.timings: Dict[str, float] = {}

        # Initialize reporting components
        self.table_formatter = TableFormatter()
        self.diagnostics = DataDiagnostics()
        self.visuals = VisualManager()
        self.report_exporter = ReportExporter()

        # PDF exporter for comprehensive report
        self.pdf_exporter = PDFReportExporter(PDFReportConfig(
            rows_per_page=40,
            include_logs=True,
            include_data_docs=True,
        ))

        # Console output capture
        self.console_output: List[str] = []

        # Run metadata
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()

    def run(self) -> int:
        """Run the pipeline with enhanced reporting."""
        steps = self.config.get_effective_steps()

        # Print grand header
        self._print_grand_header()

        total_start = time.time()

        try:
            for i, step in enumerate(steps, 1):
                self._run_step(step, i, len(steps))

            total_elapsed = time.time() - total_start

            # Print final summary
            self._print_final_summary(total_elapsed)

            # Prompt for report save
            self._finalize_report(total_elapsed)

            return 0

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return 1

    def _print_grand_header(self) -> None:
        """Print the grand pipeline header."""
        steps = self.config.get_effective_steps()

        print("\n")
        print("█" * 80)
        print("█" * 80)
        print("██")
        print("██  NBA BET BUILDER PIPELINE")
        print("██  Research-Grade Execution")
        print("██")
        print("█" * 80)
        print("█" * 80)
        print()
        print(f"  Run ID:     {self.run_id}")
        print(f"  Mode:       {self.config.mode.value.upper()}")
        print(f"  Verbosity:  {self.config.verbosity.value.upper()}")
        print(f"  Steps:      {[s.value for s in steps]}")
        print(f"  GPU:        {'Enabled' if self.config.use_gpu else 'Disabled'}")
        print(f"  Output:     {self.config.output_dir}")
        print()
        print("─" * 80)
        print()

    def _run_step(self, step: PipelineStep, step_num: int, total_steps: int) -> None:
        """Run a single pipeline step with enhanced reporting."""
        print_step_header(step.value, step_num)

        start_time = time.time()

        step_handlers = {
            PipelineStep.DATA: self._run_data_step,
            PipelineStep.DISTRIBUTIONS: self._run_distributions_step,
            PipelineStep.CORRELATIONS: self._run_correlations_step,
            PipelineStep.MONTE_CARLO: self._run_monte_carlo_step,
            PipelineStep.EDGES: self._run_edges_step,
            PipelineStep.TRAIN: self._run_train_step,
            PipelineStep.BACKTEST: self._run_backtest_step,
        }

        handler = step_handlers.get(step)
        if handler:
            result = handler()
            self.results[step.value] = result
            self.report_exporter.add_step_result(step.value, result)

        elapsed = time.time() - start_time
        self.timings[step.value] = elapsed

        print_step_footer(step.value, elapsed, result)

    def _run_data_step(self) -> dict:
        """Run data preparation with enhanced reporting."""
        from src.data_loader import (
            load_bet_builder_legs,
            load_player_game_logs,
            load_prop_lines,
        )
        from src.feature_engineering import (
            build_builder_level_dataset,
            build_leg_level_dataset,
            add_stat_composition_features,
            add_player_diversity_features,
        )
        from src.preprocessing import add_basic_context_features

        print("Loading raw data...")
        player_games = load_player_game_logs()
        prop_lines = load_prop_lines(player_games=player_games)
        bet_builder_legs = load_bet_builder_legs(
            prop_lines=prop_lines, player_games=player_games
        )

        # === DATA HEALTH DIAGNOSTICS ===
        print("\n" + "─" * 50)
        print("DATA HEALTH DIAGNOSTICS")
        print("─" * 50)

        run_data_health_checks(
            player_games, "Player Games",
            key_columns=["game_id", "player_id"],
            numeric_columns=["pts", "ast", "reb", "fg3m"]
        )

        # === TABLE: Dataset Shapes ===
        shapes_df = pd.DataFrame({
            "Dataset": ["Player Games", "Prop Lines", "Bet Builder Legs"],
            "Rows": [len(player_games), len(prop_lines), len(bet_builder_legs)],
            "Cols": [len(player_games.columns), len(prop_lines.columns), len(bet_builder_legs.columns)],
            "Memory_MB": [
                player_games.memory_usage(deep=True).sum() / 1024**2,
                prop_lines.memory_usage(deep=True).sum() / 1024**2,
                bet_builder_legs.memory_usage(deep=True).sum() / 1024**2,
            ]
        })
        print_table(shapes_df, "Dataset Shapes & Memory")

        # === TABLE: Missingness ===
        missing_data = []
        for name, df in [("Games", player_games), ("Props", prop_lines), ("Legs", bet_builder_legs)]:
            for col in df.columns[:10]:
                miss_pct = df[col].isna().sum() / len(df) * 100
                if miss_pct > 0:
                    missing_data.append({"Dataset": name, "Column": col, "Missing_%": miss_pct})

        if missing_data:
            missing_df = pd.DataFrame(missing_data).nlargest(10, "Missing_%")
            print_table(missing_df, "Top Missing Values")

        # Add context features
        print("\nAdding context features...")
        player_games = add_basic_context_features(player_games)

        # Check for zero-variance issues (like home/away)
        zero_var_cols = [col for col in player_games.columns if player_games[col].nunique() <= 1]
        if zero_var_cols:
            print_warning_block(
                [f"Zero-variance columns detected: {', '.join(zero_var_cols)}",
                 "These may be stuck flags or conversion issues."],
                title="ZERO-VARIANCE WARNING"
            )

        # Build datasets
        print("\nBuilding leg-level dataset...")
        leg_df = build_leg_level_dataset(player_games, prop_lines, bet_builder_legs)

        print("Building builder-level dataset...")
        builder_df = build_builder_level_dataset(leg_df)
        builder_df = add_stat_composition_features(leg_df, builder_df)
        builder_df = add_player_diversity_features(leg_df, builder_df)

        # === TABLE: Leg Summary ===
        leg_summary = leg_df.groupby(["stat", "direction"]).agg({
            "leg_hit": ["count", "mean"]
        }).round(3)
        leg_summary.columns = ["Count", "Hit_Rate"]
        leg_summary = leg_summary.reset_index()
        print_table(leg_summary, "Leg Hit Rates by Stat & Direction")

        # === TABLE: Builder Summary ===
        builder_summary = pd.DataFrame({
            "Metric": ["Total Builders", "Avg Legs/Builder", "Overall Hit Rate",
                       "2-Leg Builders", "3-Leg Builders", "4-Leg Builders"],
            "Value": [
                len(builder_df),
                builder_df["n_legs"].mean(),
                builder_df["builder_hit"].mean(),
                (builder_df["n_legs"] == 2).sum(),
                (builder_df["n_legs"] == 3).sum(),
                (builder_df["n_legs"] == 4).sum(),
            ]
        })
        print_table(builder_summary, "Builder Summary")

        # === VISUALS ===
        if self.config.show_plots:
            # Rest days distribution
            if "rest_days" in player_games.columns:
                rest_days = player_games["rest_days"].dropna()
                self.visuals.show_distribution(
                    rest_days.clip(upper=30),
                    title="Rest Days Distribution (Clipped at 30)",
                    data_type="rest_days",
                    filename="data_rest_days.png"
                )

            # Legs per builder
            self.visuals.show_distribution(
                builder_df["n_legs"],
                title="Legs per Builder Distribution",
                data_type="legs",
                bins=10,
                filename="data_legs_per_builder.png"
            )

            # Hit rate by stat
            hit_by_stat = leg_df.groupby("stat")["leg_hit"].mean()
            self.visuals.show_grouped_bars(
                hit_by_stat.reset_index(),
                x="stat", y="leg_hit",
                title="Hit Rate by Stat Type",
                ylabel="Hit Rate",
                filename="data_hit_rate_by_stat.png"
            )

        # Save processed data
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        player_games.to_parquet(self.config.output_dir / "player_games.parquet", index=False)
        leg_df.to_parquet(self.config.output_dir / "bet_builder_legs.parquet", index=False)
        builder_df.to_parquet(self.config.output_dir / "bet_builders.parquet", index=False)

        return {
            "n_games": len(player_games),
            "n_players": player_games["player_id"].nunique(),
            "n_legs": len(leg_df),
            "n_builders": len(builder_df),
            "leg_hit_rate": leg_df["leg_hit"].mean(),
            "builder_hit_rate": builder_df["builder_hit"].mean(),
        }

    def _run_distributions_step(self) -> dict:
        """Run distribution fitting with enhanced reporting."""
        from src.models.stat_distributions import (
            summarise_player_stat_distributions,
            attach_parametric_models,
        )

        games_path = self.config.output_dir / "player_games.parquet"
        player_games = pd.read_parquet(games_path)

        print(f"Computing distributions (min_games={self.config.min_games})...")
        summary = summarise_player_stat_distributions(
            player_games,
            stat_cols=model_config.stat_cols,
            min_games=self.config.min_games,
        )

        print("Fitting parametric models...")
        summary_with_models = attach_parametric_models(summary)

        # === TABLE: Model Family Distribution ===
        family_counts = summary_with_models["family"].value_counts()
        family_df = pd.DataFrame({
            "Family": family_counts.index,
            "Count": family_counts.values,
            "Pct": (family_counts.values / len(summary_with_models) * 100).round(1)
        })
        print_table(family_df, "Model Family Distribution")

        # === TABLE: Fitted Mean Summary by Stat ===
        mean_summary = summary_with_models.groupby("stat").agg({
            "mean": ["mean", "std", "min", "max"],
            "family": "count"
        }).round(2)
        mean_summary.columns = ["Avg_Mean", "Std_Mean", "Min_Mean", "Max_Mean", "N_Players"]
        mean_summary = mean_summary.reset_index()
        print_table(mean_summary, "Fitted Mean Summary by Stat")

        # === TABLE: Suspicious Fits (degenerate or extreme) ===
        suspicious = summary_with_models[
            (summary_with_models["family"] == "degenerate") |
            (summary_with_models["mean"] < 0.1)
        ]
        if len(suspicious) > 0:
            print_table(
                suspicious[["player_id", "stat", "family", "mean", "std"]].head(20),
                "Suspicious Fits (Degenerate or Near-Zero)"
            )

        # === VISUALS ===
        if self.config.show_plots:
            # KDE plot of fitted means per stat
            for stat in model_config.stat_cols:
                stat_data = summary_with_models[summary_with_models["stat"] == stat]["mean"]
                self.visuals.show_distribution(
                    stat_data,
                    title=f"Fitted Mean Distribution: {stat.upper()}",
                    data_type=stat,  # uses stat-specific axis labels
                    filename=f"dist_means_{stat}.png"
                )

        # Save
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        summary_with_models.to_parquet(
            self.config.models_dir / "player_stat_distributions.parquet", index=False
        )

        return {
            "n_distributions": len(summary_with_models),
            "n_players": summary_with_models["player_id"].nunique(),
            "pct_nbinom": (family_counts.get("nbinom", 0) / len(summary_with_models) * 100),
            "pct_poisson": (family_counts.get("poisson", 0) / len(summary_with_models) * 100),
            "pct_degenerate": (family_counts.get("degenerate", 0) / len(summary_with_models) * 100),
        }

    def _run_correlations_step(self) -> dict:
        """Run correlation analysis with enhanced reporting."""
        from src.models.dependencies import (
            compute_player_stat_correlations,
            compute_global_correlation,
            identify_high_correlation_pairs,
        )

        games_path = self.config.output_dir / "player_games.parquet"
        player_games = pd.read_parquet(games_path)

        print("Computing per-player correlations...")
        per_player_corr = compute_player_stat_correlations(
            player_games,
            stat_cols=model_config.stat_cols,
            min_games=self.config.min_games,
            method=model_config.correlation_method,
        )

        print("Computing global correlations...")
        global_corr = compute_global_correlation(
            player_games,
            stat_cols=model_config.stat_cols,
            method=model_config.correlation_method,
        )

        print("Identifying high correlation pairs...")
        high_corr = identify_high_correlation_pairs(per_player_corr, threshold=0.7)

        # === TABLE: Global Correlation Matrix (split if needed) ===
        # Pivot to matrix form
        corr_matrix = global_corr.pivot(index="stat_x", columns="stat_y", values="corr")
        print_table(corr_matrix.reset_index(), "Global Correlation Matrix")

        # === TABLE: Strongest Correlations ===
        strongest = global_corr.nlargest(10, "corr")[["stat_x", "stat_y", "corr"]]
        print_table(strongest, "Strongest Correlation Pairs")

        # === TABLE: High-Correlation Player Counts ===
        if len(high_corr) > 0:
            high_corr_summary = high_corr.groupby(["stat_x", "stat_y"]).size().reset_index(name="N_Players")
            high_corr_summary = high_corr_summary.sort_values("N_Players", ascending=False)
            print_table(high_corr_summary.head(10), "High Correlation Pairs (|r| >= 0.7)")

        # === VISUALS ===
        if self.config.show_plots:
            # Global correlation heatmap
            self.visuals.show_heatmap(
                corr_matrix,
                title="Global Stat Correlation Matrix",
                filename="corr_global_heatmap.png"
            )

            # Distribution of per-player correlations
            if len(per_player_corr) > 0:
                self.visuals.show_distribution(
                    per_player_corr["corr"],
                    title="Distribution of Per-Player Stat Correlations",
                    data_type="correlation",
                    filename="corr_distribution.png"
                )

        # Save
        per_player_corr.to_parquet(
            self.config.models_dir / "player_stat_correlations.parquet", index=False
        )
        global_corr.to_parquet(
            self.config.models_dir / "global_stat_correlations.parquet", index=False
        )

        return {
            "n_player_correlations": len(per_player_corr),
            "n_players_analyzed": per_player_corr["player_id"].nunique() if len(per_player_corr) > 0 else 0,
            "n_high_correlations": len(high_corr),
            "avg_correlation": per_player_corr["corr"].mean() if len(per_player_corr) > 0 else 0,
        }

    def _run_monte_carlo_step(self) -> dict:
        """Run Monte Carlo simulation with enhanced reporting and live feedback."""
        from src.models.monte_carlo_forecasting import (
            MonteCarloForecaster,
            MonteCarloConfig,
            forecast_all_players_points,
            _MLX_AVAILABLE,
        )

        # Check GPU
        backend = "GPU (Metal)" if self.config.use_gpu and _MLX_AVAILABLE else "CPU"
        print(f"Backend: {backend}")
        print(f"Simulations: {self.config.n_simulations:,}")
        print(f"Use correlations: {self.config.use_correlations}")

        # Load data
        dist_path = self.config.models_dir / "player_stat_distributions.parquet"
        corr_path = self.config.models_dir / "player_stat_correlations.parquet"

        distributions_df = pd.read_parquet(dist_path)
        correlations_df = pd.read_parquet(corr_path) if corr_path.exists() else None

        # Configure MC
        mc_config = MonteCarloConfig(
            n_simulations=self.config.n_simulations,
            random_state=self.config.random_state,
            use_gpu=self.config.use_gpu,
            use_correlations=self.config.use_correlations,
        )

        # Run forecasting
        print(f"\nForecasting {self.config.forecast_stat} for all players...")
        result_df = forecast_all_players_points(
            distributions_df,
            correlations_df,
            mc_config,
            stat=self.config.forecast_stat,
        )

        if result_df.empty:
            return {"n_forecasts": 0}

        # === TABLE: Forecast Summary Stats (split into ≤5 col tables) ===
        summary_stats = pd.DataFrame({
            "Metric": ["Mean of Means", "Mean of Stds", "Mean of Medians",
                       "Min Prediction", "Max Prediction"],
            "Value": [
                result_df["mean"].mean(),
                result_df["std"].mean(),
                result_df["median"].mean() if "median" in result_df.columns else np.nan,
                result_df["mean"].min(),
                result_df["mean"].max(),
            ]
        })
        print_table(summary_stats, "Monte Carlo Forecast Summary")

        # === TABLE: Top Predictions ===
        top_pred = result_df.nlargest(15, "mean")[["player_id", "mean", "std"]].copy()
        top_pred.columns = ["Player", "Mean", "Uncertainty"]
        print_table(top_pred, f"Top 15 Predicted {self.config.forecast_stat.upper()}")

        # === TABLE: Highest Uncertainty ===
        high_unc = result_df.nlargest(15, "std")[["player_id", "mean", "std"]].copy()
        high_unc.columns = ["Player", "Mean", "Uncertainty"]
        print_table(high_unc, "Highest Uncertainty Forecasts")

        # === VISUALS ===
        if self.config.show_plots:
            # Forecast means KDE plot
            self.visuals.show_distribution(
                result_df["mean"],
                title=f"Monte Carlo Forecast Distribution: {self.config.forecast_stat.upper()}",
                data_type="forecast",
                filename=f"mc_forecast_means_{self.config.forecast_stat}.png"
            )

            # Uncertainty KDE plot
            self.visuals.show_distribution(
                result_df["std"],
                title="Forecast Uncertainty Distribution (Standard Deviation)",
                data_type="uncertainty",
                filename="mc_forecast_uncertainty.png"
            )

        # Save
        result_df.to_parquet(self.config.models_dir / "mc_forecasts.parquet", index=False)

        return {
            "n_forecasts": len(result_df),
            "backend": backend,
            "n_simulations": self.config.n_simulations,
            "avg_prediction": float(result_df["mean"].mean()),
            "avg_uncertainty": float(result_df["std"].mean()),
        }

    def _run_edges_step(self) -> dict:
        """
        Run edge calculation with stat grouping, tiered outputs, and clear terminology.

        DATA FEEDS:
        - Player statistical distributions (fitted from historical game logs)
        - Player game logs (for name resolution)

        TERMINOLOGY:
        - "Threshold" = the statistical cutoff (formerly "line")
        - "Probability" = model-estimated chance of hitting the threshold
        - "Min Odds for +EV" = minimum decimal odds for positive expected value
        """
        from src.models.edge_calculator import (
            PlayerEdgeCalculator,
            calculate_breakeven_odds,
        )

        print("=" * 70)
        print("EDGE CALCULATION - BREAKEVEN ODDS ANALYSIS")
        print("=" * 70)
        print()
        print("DATA FEEDS INTO THIS STEP:")
        print("  1. Player Statistical Distributions (from distribution fitting)")
        print("  2. Historical Game Logs (for player name resolution)")
        print()
        print("TERMINOLOGY:")
        print("  - THRESHOLD: The statistical cutoff value for a prop bet")
        print("    (e.g., 'Over 24.5 points' = threshold of 24.5)")
        print("  - PROBABILITY: Model-estimated chance of hitting the threshold")
        print("  - MIN ODDS FOR +EV: Minimum decimal odds needed for positive edge")
        print("    Formula: Min Odds = 1 / Probability")
        print()

        dist_path = self.config.models_dir / "player_stat_distributions.parquet"
        games_path = self.config.output_dir / "player_games.parquet"

        distributions_df = pd.read_parquet(dist_path)
        player_games_df = pd.read_parquet(games_path) if games_path.exists() else None

        # Calculate edges
        edge_df = calculate_breakeven_odds(
            distributions_df,
            player_games_df,
            stats=model_config.stat_cols,
        )

        if len(edge_df) == 0:
            return {"n_edges": 0}

        # Rename 'line' to 'threshold' for clarity
        if "line" in edge_df.columns:
            edge_df = edge_df.rename(columns={"line": "threshold"})

        # Filter trivial props (very low thresholds that are almost certain)
        trivial_mask = (
            ((edge_df["stat"] == "pts") & (edge_df["threshold"] <= 2.5)) |
            ((edge_df["stat"] == "fg3m") & (edge_df["threshold"] <= 0.5)) |
            ((edge_df["stat"] == "ast") & (edge_df["threshold"] <= 0.5)) |
            ((edge_df["stat"] == "reb") & (edge_df["threshold"] <= 0.5))
        )
        trivial_df = edge_df[trivial_mask]
        main_df = edge_df[~trivial_mask]

        print(f"Filtered {len(trivial_df):,} trivial props (e.g., Over 0.5 pts)")
        print(f"Analyzing {len(main_df):,} meaningful props")
        print()

        # Stat names for display
        stat_names = {
            "pts": "POINTS",
            "fg3m": "THREE-POINTERS MADE",
            "ast": "ASSISTS",
            "reb": "REBOUNDS"
        }

        # === DISPLAY BY STAT WITH TIERED SEGMENTATION ===
        for stat in model_config.stat_cols:
            stat_df = main_df[main_df["stat"] == stat]
            if len(stat_df) == 0:
                continue

            stat_name = stat_names.get(stat, stat.upper())

            print()
            print("█" * 70)
            print(f"  {stat_name} PROPS")
            print("█" * 70)

            for direction in ["over", "under"]:
                dir_df = stat_df[stat_df["direction"] == direction].copy()
                if len(dir_df) == 0:
                    continue

                # Create tiered tables
                lower_tier, mid_tier, higher_tier = create_tiered_tables(
                    dir_df, stat, threshold_col="threshold", direction=direction
                )

                dir_label = "OVERS" if direction == "over" else "UNDERS"
                tier_interpretation = (
                    "(easier to hit)" if direction == "over" else "(harder to hit)"
                ) if True else ""

                # Display each tier (40 rows each)
                for tier_name, tier_df in [
                    ("LOWER", lower_tier),
                    ("MID", mid_tier),
                    ("HIGHER", higher_tier)
                ]:
                    if len(tier_df) == 0:
                        continue

                    # Sort by probability (best value first)
                    tier_df = tier_df.nlargest(40, "probability")

                    # Prepare display columns with clear terminology
                    display_df = tier_df[[
                        "player_name", "threshold", "probability", "min_odds_for_edge"
                    ]].copy()
                    display_df.columns = [
                        "Player",
                        "Threshold",
                        "Hit_Probability",
                        "Min_Odds_+EV"
                    ]

                    # Tier description
                    if tier_name == "LOWER":
                        tier_desc = "Lower thresholds (smaller values)"
                    elif tier_name == "MID":
                        tier_desc = "Mid-range thresholds"
                    else:
                        tier_desc = "Higher thresholds (larger values)"

                    table_title = f"{stat_name} - {dir_label} - {tier_name} TIER"

                    print_table(
                        display_df,
                        table_title,
                        show_footer=True
                    )

                    # Add to PDF exporter
                    self.pdf_exporter.add_table(
                        display_df,
                        table_title,
                        step="edges",
                        description=f"{tier_desc}. Sorted by probability (best value first)."
                    )

            # Stat summary
            print(f"\n  [{stat_name} SUMMARY]")
            print(f"    Total Candidates: {len(stat_df):,}")
            print(f"    Threshold Range: {stat_df['threshold'].min():.1f} - {stat_df['threshold'].max():.1f}")
            print(f"    Avg Hit Probability: {stat_df['probability'].mean()*100:.1f}%")
            print(f"    Avg Min Odds for +EV: {stat_df['min_odds_for_edge'].mean():.2f}")

        # === VISUALS ===
        if self.config.show_plots:
            for stat in model_config.stat_cols:
                stat_probs = main_df[main_df["stat"] == stat]["probability"]
                if len(stat_probs) > 0:
                    fig = self.visuals.show_distribution(
                        stat_probs,
                        title=f"Hit Probability Distribution: {stat_names.get(stat, stat.upper())}",
                        data_type="probability",
                        filename=f"edges_prob_dist_{stat}.png"
                    )
                    self.pdf_exporter.add_figure(fig, f"Probability Distribution: {stat.upper()}", "edges")

        # Save with updated column name
        edge_df.to_parquet(self.config.models_dir / "player_edges.parquet", index=False)

        return {
            "n_edges": len(edge_df),
            "n_meaningful": len(main_df),
            "n_trivial": len(trivial_df),
            "n_players": edge_df["player_id"].nunique(),
            "avg_probability": float(main_df["probability"].mean()),
            "avg_min_odds": float(main_df["min_odds_for_edge"].mean()),
        }

    def _run_train_step(self) -> dict:
        """Run model training with enhanced reporting."""
        import joblib
        from src.models.bet_builder_outcome import (
            train_builder_hit_model,
            default_feature_columns,
        )

        builder_path = self.config.output_dir / "bet_builders.parquet"
        builders = pd.read_parquet(builder_path)

        feature_cols = default_feature_columns()
        print(f"Training on {len(builders):,} builders")
        print(f"Features: {feature_cols}")

        model, metrics = train_builder_hit_model(
            builders,
            feature_cols=feature_cols,
            label_col="builder_hit",
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        # === TABLE: Training Metrics ===
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC AUC", "Brier Score", "Log Loss", "N Samples"],
            "Value": [
                metrics["accuracy"],
                metrics["roc_auc"],
                metrics["brier"],
                metrics["log_loss"],
                metrics["n_obs"],
            ]
        })
        print_table(metrics_df, "Training Metrics")

        # === TABLE: Feature Importance ===
        if "feature_importance" in metrics and metrics["feature_importance"] is not None:
            feat_imp = metrics["feature_importance"]
            if isinstance(feat_imp, pd.DataFrame):
                print_table(feat_imp.head(10), "Feature Importance")

        # Save model
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.config.models_dir / "builder_hit_model.joblib")

        return {
            "n_samples": metrics["n_obs"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "brier": metrics["brier"],
        }

    def _run_backtest_step(self) -> dict:
        """Run backtesting with enhanced reporting."""
        from src.backtest.simulator import run_rolling_backtest
        from src.backtest.splits import TimeSplitConfig
        from src.models.bet_builder_outcome import default_feature_columns

        builder_path = self.config.output_dir / "bet_builders.parquet"
        builders = pd.read_parquet(builder_path)

        split_cfg = TimeSplitConfig(
            date_col=backtest_config.date_col,
            train_window_days=self.config.train_window_days,
            test_window_days=self.config.test_window_days,
            min_train_size=backtest_config.min_train_size,
        )

        print(f"Train window: {split_cfg.train_window_days} days")
        print(f"Test window: {split_cfg.test_window_days} days")

        feature_cols = default_feature_columns()

        results_df, predictions_df = run_rolling_backtest(
            builders,
            split_cfg=split_cfg,
            feature_cols=feature_cols,
            label_col="builder_hit",
            random_state=self.config.random_state,
            save_predictions=self.config.save_predictions,
        )

        if len(results_df) == 0:
            return {"n_folds": 0}

        # === TABLE: Fold-by-Fold Results ===
        fold_display = results_df[["fold", "roi", "profit", "hit_rate", "brier"]].copy()
        fold_display.columns = ["Fold", "ROI", "Profit", "Hit_Rate", "Brier"]
        print_table(fold_display, "Backtest Fold Results")

        # === TABLE: Aggregate Summary ===
        overall_roi = results_df["profit"].sum() / results_df["total_stake"].sum()
        profitable_folds = (results_df["roi"] > 0).sum()

        summary_df = pd.DataFrame({
            "Metric": ["Total Folds", "Overall ROI", "Total Profit",
                       "Profitable Folds", "Avg Hit Rate"],
            "Value": [
                len(results_df),
                f"{overall_roi*100:.2f}%",
                f"{results_df['profit'].sum():.2f}",
                f"{profitable_folds}/{len(results_df)}",
                f"{results_df['hit_rate'].mean()*100:.1f}%"
            ]
        })
        print_table(summary_df, "Backtest Summary")

        # === VISUALS ===
        if self.config.show_plots and len(results_df) > 1:
            # ROI distribution across folds
            self.visuals.show_distribution(
                results_df["roi"],
                title="Return on Investment (ROI) Distribution Across Backtest Folds",
                data_type="roi",
                filename="backtest_roi_dist.png"
            )

        # Save
        results_df.to_parquet(self.config.output_dir / "backtest_results.parquet", index=False)

        return {
            "n_folds": len(results_df),
            "overall_roi": overall_roi,
            "total_profit": float(results_df["profit"].sum()),
            "profitable_folds": profitable_folds,
        }

    def _print_final_summary(self, total_elapsed: float) -> None:
        """Print the final research summary."""
        print("\n")
        print("█" * 80)
        print("██  PIPELINE COMPLETE - RESEARCH SUMMARY")
        print("█" * 80)
        print()

        # Timing summary
        print("EXECUTION TIMING:")
        print("─" * 50)
        for step, elapsed in self.timings.items():
            pct = elapsed / total_elapsed * 100
            print(f"  {step:15} {elapsed:8.2f}s ({pct:5.1f}%)")
        print(f"  {'TOTAL':15} {total_elapsed:8.2f}s")
        print()

        # Data health summary
        print("DATA HEALTH:")
        print("─" * 50)
        reports = self.diagnostics.get_all_reports()
        for report in reports:
            status = "OK" if not report.has_warnings else "WARNINGS"
            print(f"  {report.dataset_name:20} [{status}]")
        print()

        # Key metrics
        print("KEY METRICS:")
        print("─" * 50)
        for step, result in self.results.items():
            print(f"  [{step.upper()}]")
            for key, value in list(result.items())[:4]:
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        print()

        # Output locations
        print("OUTPUT LOCATIONS:")
        print("─" * 50)
        print(f"  Data:    {self.config.output_dir}")
        print(f"  Models:  {self.config.models_dir}")
        print(f"  Figures: {self.config.figures_dir}")
        print()

    def _finalize_report(self, total_elapsed: float) -> None:
        """Finalize and save the comprehensive PDF report."""
        # Set metadata for both exporters
        self.report_exporter.set_metadata(RunMetadata(
            run_id=self.run_id,
            start_time=self.start_time,
            end_time=datetime.now(),
            total_duration=total_elapsed,
            backend="GPU" if self.config.use_gpu else "CPU",
            steps_completed=list(self.timings.keys()),
        ))

        # Set overall metrics
        self.report_exporter.set_metrics({
            "total_time_s": total_elapsed,
            **{f"{k}_{mk}": mv for k, v in self.results.items()
               for mk, mv in (v.items() if isinstance(v, dict) else [])}
        })

        # Set PDF metadata
        self.pdf_exporter.set_metadata(
            run_id=self.run_id,
            start_time=self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_duration=f"{total_elapsed:.2f}s",
            backend="GPU" if self.config.use_gpu else "CPU",
            steps_completed=", ".join(self.timings.keys()),
            n_simulations=self.config.n_simulations,
        )

        # Generate PDF report (always save)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.config.figures_dir / f"pipeline_report_{timestamp}.pdf"

        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE PDF REPORT")
        print("=" * 70)
        print()
        print("The PDF report includes:")
        print("  - Data feed documentation and terminology glossary")
        print("  - All tables (40 rows each, using tabulate formatting)")
        print("  - All figures and visualizations")
        print("  - Complete console output log")
        print()

        try:
            saved_path = self.pdf_exporter.generate_pdf(pdf_path)
            print(f"PDF Report saved to: {saved_path}")
        except Exception as e:
            logger.warning(f"Failed to generate PDF: {e}")
            print(f"Warning: PDF generation failed: {e}")

        # Also offer HTML export
        prompt_save_report(self.report_exporter)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NBA Bet Builder Pipeline - Enhanced Reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "selective"],
        default="full",
        help="Execution mode",
    )

    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["lite", "full"],
        default="full",
        help="Reporting verbosity level",
    )

    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["data", "distributions", "correlations", "monte-carlo", "edges", "train", "backtest"],
        default=None,
        help="Steps to run in selective mode",
    )

    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(__import__("logging"), log_level))

    # Build config
    steps = []
    if args.steps:
        for step_name in args.steps:
            steps.append(PipelineStep(step_name))

    config = PipelineConfig(
        mode=ExecutionMode(args.mode),
        verbosity=Verbosity(args.verbosity),
        steps=steps,
        use_gpu=not args.no_gpu,
        n_simulations=args.n_sims,
        show_plots=not args.no_plots,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    runner = EnhancedPipelineRunner(config)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
