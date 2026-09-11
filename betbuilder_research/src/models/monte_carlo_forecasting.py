"""Monte Carlo simulation for player points forecasting with Apple Metal GPU acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson, norm
from scipy.linalg import cholesky
from tqdm import tqdm

from ..config import get_logger, model_config


logger = get_logger(__name__)


# Check for Metal GPU support via MLX
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
    logger.info("MLX (Apple Metal GPU) acceleration available")
except ImportError:
    logger.info("MLX not available, falling back to NumPy CPU implementation")


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_simulations: int = 10000
    random_state: int = 42
    use_gpu: bool = True  # Use Metal GPU if available
    use_correlations: bool = True  # Apply correlation structure
    batch_size: int = 5000  # Batch size for GPU processing
    confidence_levels: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# Default config
mc_config = MonteCarloConfig()


class MetalGPUSimulator:
    """
    Monte Carlo simulator using Apple Metal GPU via MLX.

    Uses MLX for fast random number generation and matrix operations
    on Apple Silicon GPUs.
    """

    def __init__(self, config: MonteCarloConfig = None):
        """
        Initialize the Metal GPU simulator.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config or mc_config

        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx"
            )

        # Set random seed
        mx.random.seed(self.config.random_state)
        logger.info(f"Metal GPU simulator initialized with {self.config.n_simulations} simulations")

    def generate_correlated_uniforms(
        self,
        n_vars: int,
        correlation_matrix: np.ndarray,
        n_sims: int = None
    ) -> np.ndarray:
        """
        Generate correlated uniform random variables using Gaussian copula on GPU.

        Args:
            n_vars: Number of variables
            correlation_matrix: Correlation matrix (n_vars x n_vars)
            n_sims: Number of simulations (defaults to config)

        Returns:
            Array of shape (n_sims, n_vars) with correlated uniforms in [0, 1]
        """
        n_sims = n_sims or self.config.n_simulations

        # Ensure correlation matrix is valid
        corr_np = np.array(correlation_matrix, dtype=np.float32)

        # Make positive semi-definite by eigenvalue adjustment
        eigenvalues, eigenvectors = np.linalg.eigh(corr_np)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        corr_np = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Cholesky decomposition
        try:
            L = cholesky(corr_np, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed, using uncorrelated samples")
            L = np.eye(n_vars, dtype=np.float32)

        # Convert to MLX arrays
        L_mx = mx.array(L)

        # Generate standard normal samples on GPU
        Z = mx.random.normal(shape=(n_sims, n_vars))

        # Apply correlation structure: X = Z @ L.T
        X = mx.matmul(Z, mx.transpose(L_mx))

        # Transform to uniform using normal CDF
        # MLX doesn't have norm.cdf, so we compute on CPU for this step
        X_np = np.array(X)
        U = norm.cdf(X_np)

        return U.astype(np.float32)

    def generate_uncorrelated_uniforms(
        self,
        n_vars: int,
        n_sims: int = None
    ) -> np.ndarray:
        """
        Generate uncorrelated uniform random variables on GPU.

        Args:
            n_vars: Number of variables
            n_sims: Number of simulations

        Returns:
            Array of shape (n_sims, n_vars) with uniforms in [0, 1]
        """
        n_sims = n_sims or self.config.n_simulations

        U = mx.random.uniform(shape=(n_sims, n_vars))

        return np.array(U, dtype=np.float32)

    def batch_inverse_transform(
        self,
        uniforms: np.ndarray,
        distributions: List[Dict],
    ) -> np.ndarray:
        """
        Apply inverse transform sampling in batches.

        Args:
            uniforms: Uniform samples of shape (n_sims, n_vars)
            distributions: List of distribution specs for each variable

        Returns:
            Simulated values of shape (n_sims, n_vars)
        """
        n_sims, n_vars = uniforms.shape
        samples = np.zeros((n_sims, n_vars), dtype=np.float32)

        for j, dist in enumerate(distributions):
            family = dist.get("family", "poisson")
            params = dist.get("params", {})

            u = uniforms[:, j]

            if family == "poisson":
                mu = params.get("mu", 1.0)
                samples[:, j] = poisson.ppf(u, mu)

            elif family == "nbinom":
                n = params.get("n", 1.0)
                p = params.get("p", 0.5)
                samples[:, j] = nbinom.ppf(u, n, p)

            elif family == "degenerate":
                loc = params.get("loc", 0)
                samples[:, j] = loc

            else:
                # Default to Poisson with mean from params
                mu = params.get("mu", params.get("mean", 10.0))
                samples[:, j] = poisson.ppf(u, max(mu, 0.1))

        return samples


class CPUSimulator:
    """
    Monte Carlo simulator using NumPy (CPU fallback).
    """

    def __init__(self, config: MonteCarloConfig = None):
        """
        Initialize the CPU simulator.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config or mc_config
        self.rng = np.random.default_rng(self.config.random_state)
        logger.info(f"CPU simulator initialized with {self.config.n_simulations} simulations")

    def generate_correlated_uniforms(
        self,
        n_vars: int,
        correlation_matrix: np.ndarray,
        n_sims: int = None
    ) -> np.ndarray:
        """Generate correlated uniforms using Gaussian copula."""
        n_sims = n_sims or self.config.n_simulations

        # Ensure valid correlation matrix
        corr_np = np.array(correlation_matrix, dtype=np.float64)

        # Make positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(corr_np)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        corr_np = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Cholesky decomposition
        try:
            L = cholesky(corr_np, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed, using uncorrelated samples")
            L = np.eye(n_vars)

        # Generate correlated normals
        Z = self.rng.standard_normal((n_sims, n_vars))
        X = Z @ L.T

        # Transform to uniforms
        U = norm.cdf(X)

        return U.astype(np.float32)

    def generate_uncorrelated_uniforms(
        self,
        n_vars: int,
        n_sims: int = None
    ) -> np.ndarray:
        """Generate uncorrelated uniforms."""
        n_sims = n_sims or self.config.n_simulations
        return self.rng.uniform(size=(n_sims, n_vars)).astype(np.float32)

    def batch_inverse_transform(
        self,
        uniforms: np.ndarray,
        distributions: List[Dict],
    ) -> np.ndarray:
        """Apply inverse transform sampling."""
        n_sims, n_vars = uniforms.shape
        samples = np.zeros((n_sims, n_vars), dtype=np.float32)

        for j, dist in enumerate(distributions):
            family = dist.get("family", "poisson")
            params = dist.get("params", {})

            u = uniforms[:, j]

            if family == "poisson":
                mu = params.get("mu", 1.0)
                samples[:, j] = poisson.ppf(u, mu)

            elif family == "nbinom":
                n = params.get("n", 1.0)
                p = params.get("p", 0.5)
                samples[:, j] = nbinom.ppf(u, n, p)

            elif family == "degenerate":
                loc = params.get("loc", 0)
                samples[:, j] = loc

            else:
                mu = params.get("mu", params.get("mean", 10.0))
                samples[:, j] = poisson.ppf(u, max(mu, 0.1))

        return samples


class MonteCarloForecaster:
    """
    Main forecaster for player points using Monte Carlo simulation.

    Supports both Metal GPU (via MLX) and CPU execution.
    """

    def __init__(
        self,
        distributions_df: pd.DataFrame,
        correlations_df: Optional[pd.DataFrame] = None,
        config: MonteCarloConfig = None
    ):
        """
        Initialize the forecaster.

        Args:
            distributions_df: Fitted distributions from stat_distributions module
                Required columns: player_id, stat, family, params, mean, std
            correlations_df: Optional correlation data from dependencies module
                Required columns: player_id, stat_x, stat_y, corr
            config: Monte Carlo configuration
        """
        self.config = config or mc_config
        self.distributions_df = distributions_df.copy()
        self.correlations_df = correlations_df.copy() if correlations_df is not None else None

        # Select simulator based on configuration and availability
        if self.config.use_gpu and _MLX_AVAILABLE:
            self.simulator = MetalGPUSimulator(self.config)
            self.backend = "metal"
        else:
            self.simulator = CPUSimulator(self.config)
            self.backend = "cpu"

        logger.info(f"MonteCarloForecaster initialized with {self.backend} backend")

        # Build lookup dictionaries
        self._build_lookups()

    def _build_lookups(self) -> None:
        """Build efficient lookup structures for distributions and correlations."""
        # Distribution lookup: (player_id, stat) -> distribution dict
        self.dist_lookup: Dict[Tuple[str, str], Dict] = {}

        for _, row in self.distributions_df.iterrows():
            key = (str(row["player_id"]), row["stat"])
            self.dist_lookup[key] = {
                "family": row["family"],
                "params": row["params"] if isinstance(row["params"], dict) else {},
                "mean": row.get("mean", 0),
                "std": row.get("std", 1),
            }

        logger.info(f"Built distribution lookup with {len(self.dist_lookup)} entries")

        # Correlation lookup: player_id -> correlation matrix for stats
        self.corr_lookup: Dict[str, pd.DataFrame] = {}

        if self.correlations_df is not None and self.config.use_correlations:
            for player_id, grp in self.correlations_df.groupby("player_id"):
                # Pivot to matrix form
                stats = sorted(set(grp["stat_x"].unique()) | set(grp["stat_y"].unique()))
                n_stats = len(stats)

                corr_matrix = np.eye(n_stats)
                stat_to_idx = {s: i for i, s in enumerate(stats)}

                for _, row in grp.iterrows():
                    i = stat_to_idx.get(row["stat_x"])
                    j = stat_to_idx.get(row["stat_y"])
                    if i is not None and j is not None:
                        corr_matrix[i, j] = row["corr"]
                        corr_matrix[j, i] = row["corr"]

                self.corr_lookup[str(player_id)] = {
                    "matrix": corr_matrix,
                    "stats": stats,
                    "stat_to_idx": stat_to_idx
                }

            logger.info(f"Built correlation lookup with {len(self.corr_lookup)} players")

    def forecast_player(
        self,
        player_id: str,
        stats: Optional[List[str]] = None,
        n_sims: int = None
    ) -> pd.DataFrame:
        """
        Generate Monte Carlo forecasts for a single player.

        Args:
            player_id: Player identifier
            stats: Stats to forecast (defaults to model_config.stat_cols)
            n_sims: Number of simulations (defaults to config)

        Returns:
            DataFrame with simulated values, shape (n_sims, len(stats))
        """
        stats = stats or model_config.stat_cols
        n_sims = n_sims or self.config.n_simulations
        player_id = str(player_id)

        # Get distributions for requested stats
        distributions = []
        valid_stats = []

        for stat in stats:
            key = (player_id, stat)
            if key in self.dist_lookup:
                distributions.append(self.dist_lookup[key])
                valid_stats.append(stat)
            else:
                logger.warning(f"No distribution found for {player_id}/{stat}")

        if not distributions:
            logger.warning(f"No valid distributions for player {player_id}")
            return pd.DataFrame()

        n_vars = len(distributions)

        # Generate uniform samples (correlated or uncorrelated)
        if (self.config.use_correlations and
            player_id in self.corr_lookup and
            n_vars > 1):

            corr_info = self.corr_lookup[player_id]

            # Build correlation matrix for requested stats
            stat_indices = [
                corr_info["stat_to_idx"].get(s)
                for s in valid_stats
            ]

            if all(i is not None for i in stat_indices):
                full_corr = corr_info["matrix"]
                sub_corr = full_corr[np.ix_(stat_indices, stat_indices)]
                uniforms = self.simulator.generate_correlated_uniforms(
                    n_vars, sub_corr, n_sims
                )
            else:
                uniforms = self.simulator.generate_uncorrelated_uniforms(n_vars, n_sims)
        else:
            uniforms = self.simulator.generate_uncorrelated_uniforms(n_vars, n_sims)

        # Apply inverse transform sampling
        samples = self.simulator.batch_inverse_transform(uniforms, distributions)

        # Return as DataFrame
        result = pd.DataFrame(samples, columns=valid_stats)
        result["player_id"] = player_id
        result["simulation_id"] = np.arange(n_sims)

        return result

    def forecast_multiple_players(
        self,
        player_ids: List[str],
        stats: Optional[List[str]] = None,
        n_sims: int = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecasts for multiple players.

        Args:
            player_ids: List of player identifiers
            stats: Stats to forecast
            n_sims: Number of simulations
            show_progress: Show progress bar

        Returns:
            DataFrame with all simulated values
        """
        results = []

        iterator = tqdm(player_ids, desc="Forecasting players") if show_progress else player_ids

        for player_id in iterator:
            player_result = self.forecast_player(player_id, stats, n_sims)
            if not player_result.empty:
                results.append(player_result)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def compute_forecast_summary(
        self,
        player_id: str,
        stat: str = "pts",
        n_sims: int = None
    ) -> Dict:
        """
        Compute summary statistics for a player's forecast.

        Args:
            player_id: Player identifier
            stat: Stat to summarize
            n_sims: Number of simulations

        Returns:
            Dictionary with forecast summary
        """
        forecast_df = self.forecast_player(player_id, [stat], n_sims)

        if forecast_df.empty or stat not in forecast_df.columns:
            return {}

        values = forecast_df[stat].values

        summary = {
            "player_id": player_id,
            "stat": stat,
            "n_simulations": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        # Add quantiles
        for q in self.config.confidence_levels:
            summary[f"p{int(q*100)}"] = float(np.percentile(values, q * 100))

        return summary

    def compute_hit_probability(
        self,
        player_id: str,
        stat: str,
        line: float,
        direction: str = "over",
        n_sims: int = None
    ) -> Dict:
        """
        Compute probability of hitting a prop line.

        Args:
            player_id: Player identifier
            stat: Stat type
            line: Prop line value
            direction: 'over' or 'under'
            n_sims: Number of simulations

        Returns:
            Dictionary with hit probability and statistics
        """
        forecast_df = self.forecast_player(player_id, [stat], n_sims)

        if forecast_df.empty or stat not in forecast_df.columns:
            return {"error": "No forecast data available"}

        values = forecast_df[stat].values

        if direction.lower() == "over":
            hits = values > line
        else:
            hits = values < line

        prob = float(np.mean(hits))

        return {
            "player_id": player_id,
            "stat": stat,
            "line": line,
            "direction": direction,
            "hit_probability": prob,
            "implied_odds": 1.0 / prob if prob > 0 else float("inf"),
            "mean_simulated": float(np.mean(values)),
            "std_simulated": float(np.std(values)),
            "n_simulations": len(values),
        }

    def batch_forecast_hit_probabilities(
        self,
        prop_lines_df: pd.DataFrame,
        n_sims: int = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Compute hit probabilities for a batch of prop lines.

        Args:
            prop_lines_df: DataFrame with columns: player_id, stat, line, direction
            n_sims: Number of simulations
            show_progress: Show progress bar

        Returns:
            DataFrame with hit probabilities for each prop
        """
        results = []

        # Group by player for efficiency
        player_groups = prop_lines_df.groupby("player_id")

        iterator = (
            tqdm(player_groups, desc="Computing hit probabilities")
            if show_progress
            else player_groups
        )

        for player_id, player_props in iterator:
            # Get all stats for this player
            stats_needed = player_props["stat"].unique().tolist()

            # Run single simulation for all stats
            forecast_df = self.forecast_player(str(player_id), stats_needed, n_sims)

            if forecast_df.empty:
                continue

            # Compute hit probability for each prop
            for _, prop in player_props.iterrows():
                stat = prop["stat"]
                line = prop["line"]
                direction = prop.get("direction", "over")

                if stat not in forecast_df.columns:
                    continue

                values = forecast_df[stat].values

                if direction.lower() == "over":
                    hits = values > line
                else:
                    hits = values < line

                prob = float(np.mean(hits))

                results.append({
                    "player_id": player_id,
                    "stat": stat,
                    "line": line,
                    "direction": direction,
                    "mc_hit_probability": prob,
                    "mc_implied_odds": 1.0 / prob if prob > 0 else float("inf"),
                    "mc_mean": float(np.mean(values)),
                    "mc_std": float(np.std(values)),
                    "mc_median": float(np.median(values)),
                })

        return pd.DataFrame(results)


def forecast_all_players_points(
    distributions_df: pd.DataFrame,
    correlations_df: Optional[pd.DataFrame] = None,
    config: MonteCarloConfig = None,
    stat: str = "pts"
) -> pd.DataFrame:
    """
    Convenience function to forecast points for all players.

    Args:
        distributions_df: Fitted distributions DataFrame
        correlations_df: Optional correlations DataFrame
        config: Monte Carlo configuration
        stat: Stat to forecast (default: 'pts')

    Returns:
        DataFrame with forecast summaries for all players
    """
    config = config or mc_config

    forecaster = MonteCarloForecaster(
        distributions_df, correlations_df, config
    )

    # Get unique players with the specified stat
    player_ids = distributions_df[
        distributions_df["stat"] == stat
    ]["player_id"].unique()

    logger.info(f"Forecasting {stat} for {len(player_ids)} players")

    results = []

    for player_id in tqdm(player_ids, desc=f"Forecasting {stat}"):
        summary = forecaster.compute_forecast_summary(str(player_id), stat)
        if summary:
            results.append(summary)

    result_df = pd.DataFrame(results)

    logger.info(f"Generated forecasts for {len(result_df)} players")

    return result_df


def compare_mc_to_parametric(
    distributions_df: pd.DataFrame,
    prop_lines_df: pd.DataFrame,
    correlations_df: Optional[pd.DataFrame] = None,
    config: MonteCarloConfig = None
) -> pd.DataFrame:
    """
    Compare Monte Carlo hit probabilities to parametric model probabilities.

    Args:
        distributions_df: Fitted distributions
        prop_lines_df: Prop lines with parametric probabilities
        correlations_df: Optional correlations
        config: Monte Carlo configuration

    Returns:
        DataFrame comparing MC vs parametric probabilities
    """
    forecaster = MonteCarloForecaster(
        distributions_df, correlations_df, config
    )

    mc_probs = forecaster.batch_forecast_hit_probabilities(prop_lines_df)

    # Merge with original prop lines
    result = prop_lines_df.merge(
        mc_probs,
        on=["player_id", "stat", "line", "direction"],
        how="left"
    )

    # Compute differences if parametric probs exist
    if "prob_over" in result.columns:
        result["prob_diff_over"] = result["mc_hit_probability"] - result["prob_over"]

    return result
