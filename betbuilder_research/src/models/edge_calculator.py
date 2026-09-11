"""
Edge Calculator Module.

Calculates break-even odds and expected value for player props based on
fitted statistical distributions. Instead of taking odds as input, this module
outputs the minimum decimal odds required for a positive edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from ..config import get_logger, model_config


logger = get_logger(__name__)


@dataclass
class EdgeResult:
    """Result of edge calculation for a single prop."""

    player_id: str
    player_name: Optional[str]
    stat: str
    line: float
    direction: str  # 'over' or 'under'
    probability: float  # True probability of hitting
    min_odds_for_edge: float  # Minimum decimal odds for +EV
    implied_prob: float  # 1 / min_odds (same as probability)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "stat": self.stat,
            "line": self.line,
            "direction": self.direction,
            "probability": self.probability,
            "min_odds_for_edge": self.min_odds_for_edge,
            "implied_prob": self.implied_prob,
        }


class PlayerEdgeCalculator:
    """
    Calculate break-even odds for player props based on fitted distributions.

    This calculator uses the player's fitted statistical distribution to compute
    the true probability of hitting a prop line, then determines what decimal
    odds are needed for a positive expected value bet.

    Example output:
        Player: LeBron James
        Stat: pts
        Line: 24.5
        Direction: over
        True Probability: 0.52
        Min Odds for +EV: 1.92

    Interpretation: Need odds of 1.92 or better to have positive edge on
    LeBron over 24.5 points.
    """

    def __init__(
        self,
        distributions_df: pd.DataFrame,
        player_names: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the edge calculator.

        Args:
            distributions_df: DataFrame with fitted distributions
                Required columns: player_id, stat, family, params, mean, std
            player_names: Optional mapping of player_id to player_name
        """
        self.distributions_df = distributions_df.copy()
        self.player_names = player_names or {}

        # Build lookup
        self._build_dist_lookup()

        logger.info(
            f"EdgeCalculator initialized with {len(self.dist_lookup)} "
            f"player-stat distributions"
        )

    def _build_dist_lookup(self) -> None:
        """Build efficient lookup for distributions."""
        self.dist_lookup: Dict[Tuple[str, str], Dict] = {}

        for _, row in self.distributions_df.iterrows():
            key = (str(row["player_id"]), row["stat"])
            self.dist_lookup[key] = {
                "family": row["family"],
                "params": row["params"] if isinstance(row["params"], dict) else {},
                "mean": row.get("mean", 0),
                "std": row.get("std", 1),
            }

    def calculate_probability(
        self,
        player_id: str,
        stat: str,
        line: float,
        direction: str = "over"
    ) -> Optional[float]:
        """
        Calculate the probability of hitting a prop line.

        Args:
            player_id: Player identifier
            stat: Stat type ('pts', 'ast', 'reb', 'fg3m')
            line: The prop line value
            direction: 'over' or 'under'

        Returns:
            Probability of hitting (0-1), or None if no distribution found
        """
        key = (str(player_id), stat)

        if key not in self.dist_lookup:
            return None

        dist_info = self.dist_lookup[key]
        family = dist_info["family"]
        params = dist_info["params"]

        # Calculate CDF at the line
        if family == "poisson":
            mu = params.get("mu", 1.0)
            # P(X > line) = 1 - P(X <= line)
            cdf_at_line = poisson.cdf(line, mu)

        elif family == "nbinom":
            n = params.get("n", 1.0)
            p = params.get("p", 0.5)
            cdf_at_line = nbinom.cdf(line, n, p)

        elif family == "degenerate":
            loc = params.get("loc", 0)
            cdf_at_line = 1.0 if line >= loc else 0.0

        else:
            # Fallback to Poisson with mean
            mu = params.get("mu", params.get("mean", 10.0))
            cdf_at_line = poisson.cdf(line, max(mu, 0.1))

        # Return probability based on direction
        if direction.lower() == "over":
            return 1.0 - cdf_at_line
        else:  # under
            return cdf_at_line

    def calculate_edge(
        self,
        player_id: str,
        stat: str,
        line: float,
        direction: str = "over"
    ) -> Optional[EdgeResult]:
        """
        Calculate the break-even odds and edge info for a prop.

        Args:
            player_id: Player identifier
            stat: Stat type
            line: The prop line value
            direction: 'over' or 'under'

        Returns:
            EdgeResult with probability and minimum odds needed for +EV
        """
        prob = self.calculate_probability(player_id, stat, line, direction)

        if prob is None:
            return None

        # Avoid division by zero
        if prob <= 0:
            min_odds = float('inf')
        elif prob >= 1:
            min_odds = 1.0
        else:
            # Break-even decimal odds = 1 / probability
            min_odds = 1.0 / prob

        player_name = self.player_names.get(str(player_id))

        return EdgeResult(
            player_id=str(player_id),
            player_name=player_name,
            stat=stat,
            line=line,
            direction=direction,
            probability=round(prob, 4),
            min_odds_for_edge=round(min_odds, 2),
            implied_prob=round(prob, 4),
        )

    def calculate_edge_at_odds(
        self,
        player_id: str,
        stat: str,
        line: float,
        direction: str,
        offered_odds: float
    ) -> Dict:
        """
        Calculate the edge given specific offered odds.

        Args:
            player_id: Player identifier
            stat: Stat type
            line: The prop line value
            direction: 'over' or 'under'
            offered_odds: The decimal odds being offered

        Returns:
            Dictionary with edge analysis
        """
        result = self.calculate_edge(player_id, stat, line, direction)

        if result is None:
            return {"error": "No distribution found"}

        # Edge = (probability * odds) - 1
        # Or: Edge = (true_prob - implied_prob) / implied_prob
        implied_prob_from_odds = 1.0 / offered_odds
        edge = (result.probability - implied_prob_from_odds)
        edge_pct = edge * 100

        ev = (result.probability * offered_odds) - 1
        ev_pct = ev * 100

        return {
            "player_id": result.player_id,
            "player_name": result.player_name,
            "stat": result.stat,
            "line": result.line,
            "direction": result.direction,
            "true_probability": result.probability,
            "min_odds_for_edge": result.min_odds_for_edge,
            "offered_odds": offered_odds,
            "implied_prob_from_odds": round(implied_prob_from_odds, 4),
            "edge": round(edge, 4),
            "edge_pct": round(edge_pct, 2),
            "ev": round(ev, 4),
            "ev_pct": round(ev_pct, 2),
            "is_positive_ev": ev > 0,
            "verdict": f"+EV ({edge_pct:.1f}%)" if ev > 0 else f"-EV ({edge_pct:.1f}%)",
        }

    def generate_player_report(
        self,
        player_id: str,
        lines: Optional[Dict[str, List[float]]] = None,
        stats: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a comprehensive edge report for a single player.

        Args:
            player_id: Player identifier
            lines: Dict mapping stat -> list of lines to analyze
                   If None, uses lines around the player's mean
            stats: Stats to analyze (defaults to model_config.stat_cols)

        Returns:
            DataFrame with edge analysis for each line
        """
        stats = stats or model_config.stat_cols
        player_id = str(player_id)

        results = []

        for stat in stats:
            key = (player_id, stat)

            if key not in self.dist_lookup:
                continue

            dist_info = self.dist_lookup[key]
            mean = dist_info["mean"]

            # Determine lines to analyze
            if lines and stat in lines:
                stat_lines = lines[stat]
            else:
                # Generate lines around the mean
                stat_lines = self._generate_lines_around_mean(mean, stat)

            for line in stat_lines:
                for direction in ["over", "under"]:
                    edge_result = self.calculate_edge(
                        player_id, stat, line, direction
                    )
                    if edge_result:
                        row = edge_result.to_dict()
                        row["mean"] = round(mean, 1)
                        row["diff_from_mean"] = round(line - mean, 1)
                        results.append(row)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Sort by stat, line, direction
        df = df.sort_values(["stat", "line", "direction"])

        return df

    def _generate_lines_around_mean(
        self,
        mean: float,
        stat: str
    ) -> List[float]:
        """Generate typical betting lines around a player's mean."""
        # Round mean to nearest 0.5
        base = round(mean * 2) / 2

        if stat == "fg3m":
            # 3-pointers have smaller ranges
            offsets = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
        elif stat in ["ast", "reb"]:
            offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        else:  # pts
            offsets = [-4.5, -2.5, -0.5, 0.5, 2.5, 4.5]

        lines = [base + offset for offset in offsets if base + offset >= 0.5]

        return sorted(set(lines))

    def generate_all_players_report(
        self,
        stats: Optional[List[str]] = None,
        min_probability: float = 0.0,
        max_min_odds: float = 10.0
    ) -> pd.DataFrame:
        """
        Generate edge report for all players in the distribution data.

        Args:
            stats: Stats to analyze
            min_probability: Filter out props with probability below this
            max_min_odds: Filter out props requiring odds above this

        Returns:
            DataFrame with edge analysis for all players
        """
        stats = stats or model_config.stat_cols

        all_results = []
        player_ids = self.distributions_df["player_id"].unique()

        logger.info(f"Generating edge report for {len(player_ids)} players...")

        for player_id in player_ids:
            player_df = self.generate_player_report(
                player_id, stats=stats
            )
            if len(player_df) > 0:
                all_results.append(player_df)

        if not all_results:
            return pd.DataFrame()

        result_df = pd.concat(all_results, ignore_index=True)

        # Apply filters
        if min_probability > 0:
            result_df = result_df[result_df["probability"] >= min_probability]

        if max_min_odds < float('inf'):
            result_df = result_df[result_df["min_odds_for_edge"] <= max_min_odds]

        logger.info(f"Generated {len(result_df)} edge calculations")

        return result_df


def calculate_breakeven_odds(
    distributions_df: pd.DataFrame,
    player_games_df: Optional[pd.DataFrame] = None,
    stats: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate break-even odds for all players based on their distributions.

    This is the main entry point for edge calculation.

    Args:
        distributions_df: Fitted distributions DataFrame
        player_games_df: Optional player games for extracting names
        stats: Stats to analyze
        output_path: Optional path to save results

    Returns:
        DataFrame with columns:
            - player_id
            - player_name (if available)
            - stat
            - line
            - direction
            - probability (true probability of hitting)
            - min_odds_for_edge (decimal odds needed for +EV)
            - mean (player's average for this stat)
            - diff_from_mean
    """
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
    result_df = calculator.generate_all_players_report(stats=stats)

    if len(result_df) > 0 and output_path:
        result_df.to_parquet(output_path, index=False)
        logger.info(f"Saved edge report to {output_path}")

    return result_df


def format_edge_summary(edge_df: pd.DataFrame, top_n: int = 20) -> str:
    """
    Format a readable summary of edge calculations.

    Args:
        edge_df: DataFrame from calculate_breakeven_odds
        top_n: Number of top edges to show

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PLAYER PROP EDGE REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Min Odds for +EV = minimum decimal odds needed for positive expected value")
    lines.append("Example: If min_odds = 1.92, you need odds >= 1.92 for +EV")
    lines.append("")

    # Group by stat
    for stat in edge_df["stat"].unique():
        stat_df = edge_df[edge_df["stat"] == stat].copy()

        lines.append("-" * 80)
        lines.append(f"STAT: {stat.upper()}")
        lines.append("-" * 80)

        # Show highest probability props (easiest to hit)
        over_df = stat_df[stat_df["direction"] == "over"].nlargest(top_n // 2, "probability")
        under_df = stat_df[stat_df["direction"] == "under"].nlargest(top_n // 2, "probability")

        lines.append("")
        lines.append(f"Top OVER props by probability:")
        for _, row in over_df.iterrows():
            name = row.get("player_name", row["player_id"])
            lines.append(
                f"  {name}: Over {row['line']} {stat} "
                f"(prob: {row['probability']:.1%}, need {row['min_odds_for_edge']:.2f}+ odds)"
            )

        lines.append("")
        lines.append(f"Top UNDER props by probability:")
        for _, row in under_df.iterrows():
            name = row.get("player_name", row["player_id"])
            lines.append(
                f"  {name}: Under {row['line']} {stat} "
                f"(prob: {row['probability']:.1%}, need {row['min_odds_for_edge']:.2f}+ odds)"
            )

        lines.append("")

    return "\n".join(lines)
