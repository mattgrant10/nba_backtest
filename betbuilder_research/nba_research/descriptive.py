"""Preserved research calculations: descriptive."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.config import get_logger
from src.visualization import (
    display_dataframe,
    display_summary_stats,
    plot_distributions,
    plot_correlation_matrix,
    plot_boxplots,
)
from .plotting import dataset_title, show_figures

logger = get_logger(__name__)


def analyze_player_level_stats(df: pd.DataFrame):
    """Detailed player-level analysis."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 3: PLAYER-LEVEL ANALYSIS")
    logger.info("=" * 100)

    # Key stats to analyze
    key_stats = ["PTS", "AST", "REB", "FG3M", "STL", "BLK", "TOV"]

    # Overall statistics
    logger.info("\n>>> Overall Player Statistics Summary:")
    display_summary_stats(df, columns=key_stats, title="Key Statistics Summary")

    # Distribution plots
    logger.info("\n>>> Plotting distributions of key statistics...")
    plot_distributions(df, columns=key_stats, bins=40, show=True)

    # Shooting percentages
    logger.info("\n>>> Shooting Percentages Summary:")
    shooting_stats = ["FG_PCT", "FG3_PCT", "FT_PCT"]
    display_summary_stats(df, columns=shooting_stats, title="Shooting Percentages")
    plot_distributions(df, columns=shooting_stats, bins=30, show=True)

    # Per-minute stats
    logger.info("\n>>> Per-Minute Statistics Summary:")
    per_min_stats = ["PTS_PER_MIN", "AST_PER_MIN", "REB_PER_MIN"]
    display_summary_stats(df, columns=per_min_stats, title="Per-Minute Statistics")

    # Top performers (by game)
    logger.info("\n>>> Top 20 Single-Game Performances (Points):")
    top_scorers = df.nlargest(20, "PTS")[
        [
            "player_name",
            "game_date_only",
            "player_team",
            "opponent_team",
            "PTS",
            "AST",
            "REB",
            "FG3M",
            "minutes_played",
            "win",
            "home",
        ]
    ]
    print("\n" + tabulate(top_scorers, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Correlation matrix
    logger.info("\n>>> Plotting correlation matrix of key statistics...")
    plot_correlation_matrix(
        df, columns=key_stats + ["minutes_played"], method="spearman", show=True
    )

    # Boxplots
    logger.info("\n>>> Creating boxplots for key statistics...")
    plot_boxplots(df, columns=key_stats[:4], show=True)

    return df


def aggregate_by_player(df: pd.DataFrame):
    """Aggregate statistics by player."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 4: PLAYER-LEVEL AGGREGATION (Season Averages)")
    logger.info("=" * 100)

    logger.info("\n>>> Aggregating statistics by player...")

    # Group by player and calculate averages
    player_agg = (
        df.groupby("player_name")
        .agg(
            {
                "personId": "first",
                "gameId": "count",  # Games played
                "PTS": "mean",
                "AST": "mean",
                "REB": "mean",
                "FG3M": "mean",
                "STL": "mean",
                "BLK": "mean",
                "TOV": "mean",
                "minutes_played": "mean",
                "FG_PCT": "mean",
                "FG3_PCT": "mean",
                "FT_PCT": "mean",
                "win": "mean",  # Win percentage
                "home": "mean",  # Home game percentage
                "PLUS_MINUS": "mean",
            }
        )
        .round(2)
    )

    # Rename columns
    player_agg.columns = [
        "player_id",
        "games_played",
        "PPG",
        "APG",
        "RPG",
        "FG3M_PG",
        "SPG",
        "BPG",
        "TPG",
        "MPG",
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
        "WIN_PCT",
        "HOME_PCT",
        "AVG_PLUS_MINUS",
    ]

    # Reset index
    player_agg = player_agg.reset_index()

    # Filter players with at least 10 games
    logger.info("\n>>> Filtering players with at least 10 games played...")
    initial_players = len(player_agg)
    player_agg = player_agg[player_agg["games_played"] >= 10].copy()
    logger.info(f"    Players with ≥10 games: {len(player_agg):,} / {initial_players:,}")

    # Calculate additional metrics
    logger.info("\n>>> Calculating composite metrics...")
    player_agg["PTS_AST"] = player_agg["PPG"] + player_agg["APG"]
    player_agg["PTS_REB"] = player_agg["PPG"] + player_agg["RPG"]
    player_agg["PTS_AST_REB"] = player_agg["PPG"] + player_agg["APG"] + player_agg["RPG"]

    # Sort by PPG
    player_agg = player_agg.sort_values("PPG", ascending=False)

    display_dataframe(player_agg, "Player Season Averages", max_rows=25)
    display_summary_stats(player_agg, title="Player Season Averages - Summary Statistics")

    # Top players
    logger.info("\n>>> Top 20 Players by Points Per Game:")
    top_ppg = player_agg.nlargest(20, "PPG")[
        ["player_name", "games_played", "PPG", "APG", "RPG", "FG3M_PG", "FG_PCT", "MPG", "WIN_PCT"]
    ]
    print("\n" + tabulate(top_ppg, headers="keys", tablefmt="grid", showindex=False))
    print()

    logger.info("\n>>> Top 20 Players by Assists Per Game:")
    top_apg = player_agg.nlargest(20, "APG")[
        ["player_name", "games_played", "APG", "PPG", "RPG", "MPG", "WIN_PCT"]
    ]
    print("\n" + tabulate(top_apg, headers="keys", tablefmt="grid", showindex=False))
    print()

    logger.info("\n>>> Top 20 Players by Rebounds Per Game:")
    top_rpg = player_agg.nlargest(20, "RPG")[
        ["player_name", "games_played", "RPG", "PPG", "APG", "MPG", "WIN_PCT"]
    ]
    print("\n" + tabulate(top_rpg, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Plot distributions of averages
    logger.info("\n>>> Plotting distributions of player averages...")
    plot_distributions(player_agg, columns=["PPG", "APG", "RPG", "FG3M_PG"], bins=30, show=True)

    # Shooting efficiency
    logger.info("\n>>> Top 20 Most Efficient Shooters (FG% with ≥10 PPG):")
    efficient_shooters = player_agg[player_agg["PPG"] >= 10].nlargest(20, "FG_PCT")[
        ["player_name", "games_played", "PPG", "FG_PCT", "FG3_PCT", "FT_PCT"]
    ]
    print("\n" + tabulate(efficient_shooters, headers="keys", tablefmt="grid", showindex=False))
    print()

    return player_agg


def aggregate_by_team(df: pd.DataFrame):
    """Aggregate statistics by team."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 5: TEAM-LEVEL AGGREGATION")
    logger.info("=" * 100)

    logger.info("\n>>> Aggregating statistics by team...")

    team_agg = (
        df.groupby("player_team")
        .agg(
            {
                "gameId": "count",
                "PTS": "mean",
                "AST": "mean",
                "REB": "mean",
                "FG3M": "mean",
                "STL": "mean",
                "BLK": "mean",
                "TOV": "mean",
                "FG_PCT": "mean",
                "FG3_PCT": "mean",
                "win": "mean",
                "PLUS_MINUS": "mean",
                "personId": "nunique",  # Unique players
            }
        )
        .round(2)
    )

    team_agg.columns = [
        "total_player_games",
        "avg_PTS",
        "avg_AST",
        "avg_REB",
        "avg_3PM",
        "avg_STL",
        "avg_BLK",
        "avg_TOV",
        "avg_FG_PCT",
        "avg_3P_PCT",
        "win_rate",
        "avg_plus_minus",
        "unique_players",
    ]

    team_agg = team_agg.sort_values("win_rate", ascending=False).reset_index()

    display_dataframe(team_agg, "Team Statistics", max_rows=30)

    logger.info("\n>>> All Teams Ranked by Win Rate:")
    print("\n" + tabulate(team_agg, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Plot team statistics
    logger.info("\n>>> Plotting team statistics distributions...")
    plot_distributions(
        team_agg, columns=["avg_PTS", "avg_AST", "avg_REB", "win_rate"], bins=15, show=True
    )

    return team_agg


def analyze_home_away_performance(df: pd.DataFrame):
    """Analyze home vs away performance."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 6: HOME VS AWAY PERFORMANCE ANALYSIS")
    logger.info("=" * 100)

    logger.info("\n>>> Comparing home vs away statistics...")

    home_away_stats = (
        df.groupby("home")
        .agg(
            {
                "gameId": "count",
                "PTS": "mean",
                "AST": "mean",
                "REB": "mean",
                "FG_PCT": "mean",
                "FG3_PCT": "mean",
                "win": "mean",
                "PLUS_MINUS": "mean",
            }
        )
        .round(3)
    )

    home_away_stats.index = ["Away", "Home"]
    home_away_stats.columns = [
        "Games",
        "Avg PTS",
        "Avg AST",
        "Avg REB",
        "FG%",
        "3P%",
        "Win Rate",
        "Avg +/-",
    ]

    logger.info("\n>>> Home vs Away Comparison:")
    print("\n" + tabulate(home_away_stats, headers="keys", tablefmt="grid"))
    print()

    # Calculate differences
    if len(home_away_stats) == 2:
        diff = home_away_stats.loc["Home"] - home_away_stats.loc["Away"]
        logger.info("\n>>> Home Advantage (Home - Away):")
        print("\n" + tabulate(diff.to_frame("Difference"), headers="keys", tablefmt="grid"))
        print()

    # Visualize
    logger.info("\n>>> Plotting home vs away distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (stat, ax) in enumerate(zip(["PTS", "AST", "REB", "FG_PCT"], axes.flatten())):
        df.boxplot(column=stat, by="home", ax=ax)
        ax.set_title(dataset_title(f"{stat} - Home vs Away"))
        ax.set_xlabel("Home (0=Away, 1=Home)")
        ax.set_ylabel(stat)

    plt.tight_layout()
    show_figures()

    return home_away_stats


def analyze_win_loss_performance(df: pd.DataFrame):
    """Analyze performance in wins vs losses."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 7: WIN VS LOSS PERFORMANCE ANALYSIS")
    logger.info("=" * 100)

    logger.info("\n>>> Comparing statistics in wins vs losses...")

    win_loss_stats = (
        df.groupby("win")
        .agg(
            {
                "gameId": "count",
                "PTS": "mean",
                "AST": "mean",
                "REB": "mean",
                "FG_PCT": "mean",
                "FG3_PCT": "mean",
                "TOV": "mean",
                "PLUS_MINUS": "mean",
                "STL": "mean",
                "BLK": "mean",
            }
        )
        .round(3)
    )

    win_loss_stats.index = ["Loss", "Win"]
    win_loss_stats.columns = [
        "Games",
        "Avg PTS",
        "Avg AST",
        "Avg REB",
        "FG%",
        "3P%",
        "Avg TOV",
        "Avg +/-",
        "Avg STL",
        "Avg BLK",
    ]

    logger.info("\n>>> Win vs Loss Comparison:")
    print("\n" + tabulate(win_loss_stats, headers="keys", tablefmt="grid"))
    print()

    # Calculate differences
    if len(win_loss_stats) == 2:
        diff = win_loss_stats.loc["Win"] - win_loss_stats.loc["Loss"]
        logger.info("\n>>> Win Impact (Win - Loss):")
        print("\n" + tabulate(diff.to_frame("Difference"), headers="keys", tablefmt="grid"))
        print()

    return win_loss_stats


def analyze_temporal_trends(df: pd.DataFrame):
    """Analyze trends over time."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 8: TEMPORAL TRENDS ANALYSIS")
    logger.info("=" * 100)

    # By day of week
    logger.info("\n>>> Performance by Day of Week:")
    day_stats = (
        df.groupby("day_of_week")
        .agg({"gameId": "count", "PTS": "mean", "AST": "mean", "REB": "mean", "win": "mean"})
        .round(2)
    )

    day_stats.columns = ["Games", "Avg PTS", "Avg AST", "Avg REB", "Win Rate"]

    # Reorder by day
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_stats = day_stats.reindex([d for d in day_order if d in day_stats.index])

    print("\n" + tabulate(day_stats, headers="keys", tablefmt="grid"))
    print()

    # By date (time series)
    logger.info("\n>>> Daily aggregated statistics over time...")
    daily_stats = (
        df.groupby("game_date_only")
        .agg({"gameId": "count", "PTS": "mean", "AST": "mean", "REB": "mean", "win": "mean"})
        .reset_index()
    )

    daily_stats.columns = ["Date", "Games", "Avg PTS", "Avg AST", "Avg REB", "Win Rate"]
    display_dataframe(daily_stats, "Daily Statistics Over Time", max_rows=20)

    # Plot time series
    logger.info("\n>>> Plotting temporal trends...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(daily_stats["Date"], daily_stats["Avg PTS"], marker="o", markersize=3)
    axes[0, 0].set_title(dataset_title("Average Points Over Time"))
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Points")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].plot(
        daily_stats["Date"], daily_stats["Avg AST"], marker="o", markersize=3, color="orange"
    )
    axes[0, 1].set_title(dataset_title("Average Assists Over Time"))
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Assists")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].plot(
        daily_stats["Date"], daily_stats["Avg REB"], marker="o", markersize=3, color="green"
    )
    axes[1, 0].set_title(dataset_title("Average Rebounds Over Time"))
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Rebounds")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].plot(
        daily_stats["Date"], daily_stats["Win Rate"], marker="o", markersize=3, color="red"
    )
    axes[1, 1].set_title(dataset_title("Win Rate Over Time"))
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Win Rate")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    show_figures()

    return daily_stats
