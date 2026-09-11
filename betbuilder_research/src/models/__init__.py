"""Statistical models for player performance analysis."""

from .stat_distributions import (
    summarise_player_stat_distributions,
    attach_parametric_models,
    fit_parametric_count_model,
    compute_tail_probabilities,
    compute_value_bets,
)

from .dependencies import (
    compute_player_stat_correlations,
    compute_global_correlation,
    analyze_correlation_patterns,
    identify_high_correlation_pairs,
)

from .monte_carlo_forecasting import (
    MonteCarloForecaster,
    MonteCarloConfig,
    MetalGPUSimulator,
    CPUSimulator,
    forecast_all_players_points,
    compare_mc_to_parametric,
)

from .edge_calculator import (
    PlayerEdgeCalculator,
    EdgeResult,
    calculate_breakeven_odds,
    format_edge_summary,
)

from .line_generator import (
    LearnedLineGenerator,
    LineGeneratorMetrics,
    train_line_generator,
    generate_synthetic_lines,
    validate_real_lines_csv,
    load_real_lines_csv,
    compute_line_features,
    american_to_decimal,
    transform_wide_lines_to_long,
    load_wide_format_lines,
)

__all__ = [
    # Distributions
    "summarise_player_stat_distributions",
    "attach_parametric_models",
    "fit_parametric_count_model",
    "compute_tail_probabilities",
    "compute_value_bets",
    # Dependencies
    "compute_player_stat_correlations",
    "compute_global_correlation",
    "analyze_correlation_patterns",
    "identify_high_correlation_pairs",
    # Monte Carlo
    "MonteCarloForecaster",
    "MonteCarloConfig",
    "MetalGPUSimulator",
    "CPUSimulator",
    "forecast_all_players_points",
    "compare_mc_to_parametric",
    # Edge Calculator
    "PlayerEdgeCalculator",
    "EdgeResult",
    "calculate_breakeven_odds",
    "format_edge_summary",
    # Line Generator
    "LearnedLineGenerator",
    "LineGeneratorMetrics",
    "train_line_generator",
    "generate_synthetic_lines",
    "validate_real_lines_csv",
    "load_real_lines_csv",
    "compute_line_features",
    "american_to_decimal",
    "transform_wide_lines_to_long",
    "load_wide_format_lines",
]
