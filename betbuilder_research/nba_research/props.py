"""Preserved research calculations: props."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.config import get_logger
from .plotting import dataset_title, show_figures

logger = get_logger(__name__)


def identify_prop_bet_opportunities(df: pd.DataFrame, player_agg: pd.DataFrame):
    """Identify potential prop bet opportunities based on consistency."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 9: PROP BET OPPORTUNITY IDENTIFICATION")
    logger.info("=" * 100)

    logger.info("\n>>> Analyzing player consistency for prop betting...")

    # Calculate std deviation for each player
    player_std = (
        df.groupby("player_name")
        .agg({"PTS": "std", "AST": "std", "REB": "std", "FG3M": "std", "gameId": "count"})
        .round(2)
    )

    player_std.columns = ["PTS_STD", "AST_STD", "REB_STD", "FG3M_STD", "games"]

    # Merge with averages
    prop_analysis = player_agg.merge(
        player_std, left_on="player_name", right_index=True, how="inner"
    )

    # Calculate coefficient of variation (lower = more consistent)
    prop_analysis["PTS_CV"] = (prop_analysis["PTS_STD"] / prop_analysis["PPG"]).round(3)
    prop_analysis["AST_CV"] = (
        (prop_analysis["AST_STD"] / prop_analysis["APG"])
        .replace([np.inf, -np.inf], np.nan)
        .round(3)
    )
    prop_analysis["REB_CV"] = (prop_analysis["REB_STD"] / prop_analysis["RPG"]).round(3)
    prop_analysis["FG3M_CV"] = (
        (prop_analysis["FG3M_STD"] / prop_analysis["FG3M_PG"])
        .replace([np.inf, -np.inf], np.nan)
        .round(3)
    )

    # Filter for relevant players (≥20 games, ≥15 PPG)
    prop_candidates = prop_analysis[
        (prop_analysis["games_played"] >= 20) & (prop_analysis["PPG"] >= 15)
    ].copy()

    logger.info(f"\n>>> Found {len(prop_candidates)} players meeting criteria (≥20 games, ≥15 PPG)")

    # Most consistent scorers (low CV)
    logger.info("\n>>> Top 15 Most Consistent Scorers (Low Points CV):")
    consistent_scorers = prop_candidates.nlargest(15, "PPG").nsmallest(15, "PTS_CV")[
        ["player_name", "games_played", "PPG", "PTS_STD", "PTS_CV", "WIN_PCT"]
    ]
    print("\n" + tabulate(consistent_scorers, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Most consistent assist players
    logger.info("\n>>> Top 15 Most Consistent Assist Players (Low Assists CV, ≥5 APG):")
    consistent_assists = prop_candidates[prop_candidates["APG"] >= 5].nsmallest(15, "AST_CV")[
        ["player_name", "games_played", "APG", "AST_STD", "AST_CV", "WIN_PCT"]
    ]
    print("\n" + tabulate(consistent_assists, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Most consistent rebounders
    logger.info("\n>>> Top 15 Most Consistent Rebounders (Low Rebounds CV, ≥8 RPG):")
    consistent_rebounds = prop_candidates[prop_candidates["RPG"] >= 8].nsmallest(15, "REB_CV")[
        ["player_name", "games_played", "RPG", "REB_STD", "REB_CV", "WIN_PCT"]
    ]
    print("\n" + tabulate(consistent_rebounds, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Most consistent three-point shooters
    logger.info("\n>>> Top 15 Most Consistent Three-Point Shooters (Low 3PM CV, ≥2 3PM/G):")
    consistent_threes = prop_candidates[prop_candidates["FG3M_PG"] >= 2].nsmallest(15, "FG3M_CV")[
        ["player_name", "games_played", "FG3M_PG", "FG3M_STD", "FG3M_CV", "WIN_PCT"]
    ]
    print("\n" + tabulate(consistent_threes, headers="keys", tablefmt="grid", showindex=False))
    print()

    # Scatter plots: Average vs Consistency (now including 3-pointers)
    logger.info("\n>>> Plotting consistency analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Points
    axes[0, 0].scatter(prop_candidates["PPG"], prop_candidates["PTS_CV"], alpha=0.6, s=80)
    axes[0, 0].set_xlabel("Points Per Game")
    axes[0, 0].set_ylabel("Coefficient of Variation (lower = more consistent)")
    axes[0, 0].set_title(dataset_title("Scoring Consistency vs Average"))
    axes[0, 0].grid(True, alpha=0.3)

    # Assists
    axes[0, 1].scatter(
        prop_candidates["APG"], prop_candidates["AST_CV"], alpha=0.6, s=80, color="orange"
    )
    axes[0, 1].set_xlabel("Assists Per Game")
    axes[0, 1].set_ylabel("Coefficient of Variation (lower = more consistent)")
    axes[0, 1].set_title(dataset_title("Assist Consistency vs Average"))
    axes[0, 1].grid(True, alpha=0.3)

    # Rebounds
    axes[1, 0].scatter(
        prop_candidates["RPG"], prop_candidates["REB_CV"], alpha=0.6, s=80, color="green"
    )
    axes[1, 0].set_xlabel("Rebounds Per Game")
    axes[1, 0].set_ylabel("Coefficient of Variation (lower = more consistent)")
    axes[1, 0].set_title(dataset_title("Rebound Consistency vs Average"))
    axes[1, 0].grid(True, alpha=0.3)

    # Three-Pointers
    axes[1, 1].scatter(
        prop_candidates["FG3M_PG"], prop_candidates["FG3M_CV"], alpha=0.6, s=80, color="red"
    )
    axes[1, 1].set_xlabel("Three-Pointers Per Game")
    axes[1, 1].set_ylabel("Coefficient of Variation (lower = more consistent)")
    axes[1, 1].set_title(dataset_title("Three-Point Consistency vs Average"))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    show_figures()

    # Team-by-team consistency analysis
    logger.info("\n" + "=" * 100)
    logger.info("TEAM-BY-TEAM CONSISTENCY ANALYSIS")
    logger.info("=" * 100)
    logger.info("\n>>> Creating team-by-team scatterplots for consistency vs performance...")

    # Get team for each player (most common team they played for)
    player_teams = (
        df[["player_name", "player_team"]]
        .groupby("player_name")["player_team"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )
    player_teams.columns = ["player_name", "team"]

    # Merge team info into prop_analysis
    prop_with_team = prop_analysis.merge(player_teams, on="player_name", how="left")

    # Get all unique teams
    teams = sorted(prop_with_team["team"].dropna().unique())
    logger.info(f">>> Analyzing {len(teams)} teams")

    # Create INDIVIDUAL team plots for POINTS - one plot per team with ALL player names
    logger.info("\n>>> Creating Points Consistency by Team plots (individual per team)...")
    n_teams = len(teams)

    for idx, team in enumerate(teams):
        team_data = prop_with_team[
            (prop_with_team["team"] == team) & (prop_with_team["games_played"] >= 10)
        ].copy()

        if len(team_data) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Color by consistency
            colors = [
                "green" if cv < 0.40 else "orange" if cv < 0.60 else "red"
                for cv in team_data["PTS_CV"]
            ]

            ax.scatter(
                team_data["PPG"],
                team_data["PTS_CV"],
                alpha=0.7,
                s=150,
                c=colors,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add ALL player labels
            for _, player in team_data.iterrows():
                ax.annotate(
                    player["player_name"].split()[-1],  # Last name only
                    (player["PPG"], player["PTS_CV"]),
                    fontsize=9,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            ax.axhline(
                0.40,
                color="blue",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Good consistency (CV=0.40)",
            )
            ax.set_xlabel("Points Per Game", fontsize=11, fontweight="bold")
            ax.set_ylabel("CV (lower = more consistent)", fontsize=11, fontweight="bold")
            ax.set_title(
                dataset_title(f"{team} - Points Consistency vs PPG\n({len(team_data)} players)"),
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            plt.tight_layout()
            show_figures()

        if (idx + 1) % 10 == 0:
            logger.info(f"    Completed {idx + 1}/{n_teams} teams (Points)")

    # Create INDIVIDUAL team plots for ASSISTS - one plot per team with ALL player names
    logger.info("\n>>> Creating Assists Consistency by Team plots (individual per team)...")

    for idx, team in enumerate(teams):
        team_data = prop_with_team[
            (prop_with_team["team"] == team)
            & (prop_with_team["games_played"] >= 10)
            & (prop_with_team["APG"] >= 1)  # Filter for relevant assist players
        ].copy()

        if len(team_data) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Color by consistency
            colors = [
                "green" if cv < 0.40 else "orange" if cv < 0.60 else "red"
                for cv in team_data["AST_CV"]
            ]

            ax.scatter(
                team_data["APG"],
                team_data["AST_CV"],
                alpha=0.7,
                s=150,
                c=colors,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add ALL player labels
            for _, player in team_data.iterrows():
                ax.annotate(
                    player["player_name"].split()[-1],  # Last name only
                    (player["APG"], player["AST_CV"]),
                    fontsize=9,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            ax.axhline(
                0.40,
                color="blue",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Good consistency (CV=0.40)",
            )
            ax.set_xlabel("Assists Per Game", fontsize=11, fontweight="bold")
            ax.set_ylabel("CV (lower = more consistent)", fontsize=11, fontweight="bold")
            ax.set_title(
                dataset_title(f"{team} - Assists Consistency vs APG\n({len(team_data)} players)"),
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            plt.tight_layout()
            show_figures()

        if (idx + 1) % 10 == 0:
            logger.info(f"    Completed {idx + 1}/{n_teams} teams (Assists)")

    # Create INDIVIDUAL team plots for REBOUNDS - one plot per team with ALL player names
    logger.info("\n>>> Creating Rebounds Consistency by Team plots (individual per team)...")

    for idx, team in enumerate(teams):
        team_data = prop_with_team[
            (prop_with_team["team"] == team)
            & (prop_with_team["games_played"] >= 10)
            & (prop_with_team["RPG"] >= 2)  # Filter for relevant rebounders
        ].copy()

        if len(team_data) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Color by consistency
            colors = [
                "green" if cv < 0.40 else "orange" if cv < 0.60 else "red"
                for cv in team_data["REB_CV"]
            ]

            ax.scatter(
                team_data["RPG"],
                team_data["REB_CV"],
                alpha=0.7,
                s=150,
                c=colors,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add ALL player labels
            for _, player in team_data.iterrows():
                ax.annotate(
                    player["player_name"].split()[-1],  # Last name only
                    (player["RPG"], player["REB_CV"]),
                    fontsize=9,
                    fontweight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            ax.axhline(
                0.40,
                color="blue",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Good consistency (CV=0.40)",
            )
            ax.set_xlabel("Rebounds Per Game", fontsize=11, fontweight="bold")
            ax.set_ylabel("CV (lower = more consistent)", fontsize=11, fontweight="bold")
            ax.set_title(
                dataset_title(f"{team} - Rebounds Consistency vs RPG\n({len(team_data)} players)"),
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            plt.tight_layout()
            show_figures()

        if (idx + 1) % 10 == 0:
            logger.info(f"    Completed {idx + 1}/{n_teams} teams (Rebounds)")

    logger.info("\n✓ Team-by-team consistency analysis complete!")

    # Display consistency tables per team (paginated)
    logger.info("\n" + "=" * 100)
    logger.info("TEAM CONSISTENCY TABLES")
    logger.info("=" * 100)
    logger.info("\n>>> Generating consistency tables for each team...")
    logger.info(f">>> Total teams: {len(teams)}")
    logger.info(">>> Tables will be shown in batches of 5 for readability")

    # Paginate teams - show 5 at a time
    teams_per_page = 5
    total_pages = (len(teams) + teams_per_page - 1) // teams_per_page

    for page_num in range(total_pages):
        start_idx = page_num * teams_per_page
        end_idx = min((page_num + 1) * teams_per_page, len(teams))
        teams_on_page = teams[start_idx:end_idx]

        print("\n" + "=" * 100)
        print(
            f"PAGE {page_num + 1} of {total_pages} - Teams {start_idx + 1}-{end_idx} of {len(teams)}"
        )
        print("=" * 100)

        for team in teams_on_page:
            team_data = prop_with_team[
                (prop_with_team["team"] == team) & (prop_with_team["games_played"] >= 10)
            ].copy()

            if len(team_data) > 0:
                # Select relevant columns and sort by PPG
                consistency_table = (
                    team_data[
                        [
                            "player_name",
                            "games_played",
                            "PPG",
                            "PTS_CV",
                            "APG",
                            "AST_CV",
                            "RPG",
                            "REB_CV",
                            "FG3M_PG",
                            "FG3M_CV",
                        ]
                    ]
                    .sort_values("PPG", ascending=False)
                    .round(2)
                )

                # Rename columns for clarity
                consistency_table.columns = [
                    "Player",
                    "Games",
                    "PPG",
                    "PTS_CV",
                    "APG",
                    "AST_CV",
                    "RPG",
                    "REB_CV",
                    "3PM",
                    "3PM_CV",
                ]

                logger.info("\n" + "-" * 100)
                logger.info(f"{team} - CONSISTENCY METRICS")
                logger.info("-" * 100)
                logger.info(f"Players: {len(team_data)} | Games played: ≥10")
                logger.info("CV = Coefficient of Variation (Lower = More Consistent)")

                print(
                    "\n"
                    + tabulate(
                        consistency_table,
                        headers="keys",
                        tablefmt="grid",
                        showindex=False,
                        floatfmt=".2f",
                    )
                )
                print()

        # Pause between pages (except on last page)
        if page_num < total_pages - 1:
            print("\n" + "=" * 100)
            print(f"End of Page {page_num + 1}/{total_pages}")
            print("=" * 100 + "\n")

            # Small pause to prevent overwhelming output
            import time

            time.sleep(0.5)

    logger.info("\n✓ All team consistency tables displayed!")

    # SIMULATION/TESTING SECTION - High Consistency Players
    logger.info("\n" + "=" * 100)
    logger.info("PROP BET SIMULATION - HIGH CONSISTENCY PLAYERS")
    logger.info("=" * 100)
    logger.info("\n>>> Identifying players with CV < 0.40 for all stat categories...")
    logger.info(">>> These players are the most reliable for prop betting")

    # Filter for highly consistent players across all categories
    high_consistency = prop_analysis[
        (prop_analysis["games_played"] >= 15)
        & (prop_analysis["PTS_CV"] < 0.40)
        & (prop_analysis["AST_CV"] < 0.40)
        & (prop_analysis["REB_CV"] < 0.40)
        & (prop_analysis["FG3M_CV"] < 0.40)
    ].copy()

    logger.info(
        f"\n>>> Found {len(high_consistency)} players with CV < 0.40 across ALL categories (PTS, AST, REB, 3PM)"
    )

    if len(high_consistency) > 0:
        logger.info("\n>>> ELITE CONSISTENCY - All Stats CV < 0.40:")
        elite_table = (
            high_consistency[
                [
                    "player_name",
                    "games_played",
                    "PPG",
                    "PTS_CV",
                    "APG",
                    "AST_CV",
                    "RPG",
                    "REB_CV",
                    "FG3M_PG",
                    "FG3M_CV",
                    "WIN_PCT",
                ]
            ]
            .sort_values("PPG", ascending=False)
            .round(2)
        )

        elite_table.columns = [
            "Player",
            "Games",
            "PPG",
            "PTS_CV",
            "APG",
            "AST_CV",
            "RPG",
            "REB_CV",
            "3PM",
            "3PM_CV",
            "Win%",
        ]

        print(
            "\n"
            + tabulate(
                elite_table, headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f"
            )
        )
        print()
    else:
        logger.info("\n>>> No players found with CV < 0.40 across ALL categories")
        logger.info(">>> Relaxing criteria...")

    # Also show players with CV < 0.40 for Points only
    logger.info("\n>>> POINTS CONSISTENCY - PTS_CV < 0.40:")
    pts_consistent = (
        prop_analysis[
            (prop_analysis["games_played"] >= 15)
            & (prop_analysis["PPG"] >= 10)
            & (prop_analysis["PTS_CV"] < 0.40)
        ]
        .sort_values("PPG", ascending=False)[
            ["player_name", "games_played", "PPG", "PTS_STD", "PTS_CV", "WIN_PCT"]
        ]
        .head(20)
        .round(2)
    )

    pts_consistent.columns = ["Player", "Games", "PPG", "PTS_STD", "PTS_CV", "Win%"]
    print(
        "\n"
        + tabulate(pts_consistent, headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f")
    )
    print()

    # Assists consistency
    logger.info("\n>>> ASSISTS CONSISTENCY - AST_CV < 0.40:")
    ast_consistent = (
        prop_analysis[
            (prop_analysis["games_played"] >= 15)
            & (prop_analysis["APG"] >= 3)
            & (prop_analysis["AST_CV"] < 0.40)
        ]
        .sort_values("APG", ascending=False)[
            ["player_name", "games_played", "APG", "AST_STD", "AST_CV", "WIN_PCT"]
        ]
        .head(20)
        .round(2)
    )

    ast_consistent.columns = ["Player", "Games", "APG", "AST_STD", "AST_CV", "Win%"]
    print(
        "\n"
        + tabulate(ast_consistent, headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f")
    )
    print()

    # Rebounds consistency
    logger.info("\n>>> REBOUNDS CONSISTENCY - REB_CV < 0.40:")
    reb_consistent = (
        prop_analysis[
            (prop_analysis["games_played"] >= 15)
            & (prop_analysis["RPG"] >= 5)
            & (prop_analysis["REB_CV"] < 0.40)
        ]
        .sort_values("RPG", ascending=False)[
            ["player_name", "games_played", "RPG", "REB_STD", "REB_CV", "WIN_PCT"]
        ]
        .head(20)
        .round(2)
    )

    reb_consistent.columns = ["Player", "Games", "RPG", "REB_STD", "REB_CV", "Win%"]
    print(
        "\n"
        + tabulate(reb_consistent, headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f")
    )
    print()

    # Three-pointers consistency
    logger.info("\n>>> THREE-POINTERS CONSISTENCY - FG3M_CV < 0.40:")
    fg3_consistent = (
        prop_analysis[
            (prop_analysis["games_played"] >= 15)
            & (prop_analysis["FG3M_PG"] >= 2)
            & (prop_analysis["FG3M_CV"] < 0.40)
        ]
        .sort_values("FG3M_PG", ascending=False)[
            ["player_name", "games_played", "FG3M_PG", "FG3M_STD", "FG3M_CV", "WIN_PCT"]
        ]
        .head(20)
        .round(2)
    )

    fg3_consistent.columns = ["Player", "Games", "3PM", "3PM_STD", "3PM_CV", "Win%"]
    print(
        "\n"
        + tabulate(fg3_consistent, headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f")
    )
    print()

    logger.info("\n✓ Simulation analysis complete - prop bet targets identified!")

    return prop_analysis


def create_table_figure(data: pd.DataFrame, title: str, subtitle: str = "") -> plt.Figure:
    """
    Create a matplotlib figure from a pandas DataFrame table.

    Args:
        data: DataFrame to display
        title: Main title
        subtitle: Optional subtitle

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, max(6, len(data) * 0.4 + 2)))
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header styling
    for i in range(len(data.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(len(data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F2F2F2")
            else:
                table[(i, j)].set_facecolor("white")

    # Add title
    title_text = dataset_title(title)
    if subtitle:
        title_text += f"\n{subtitle}"

    fig.text(0.5, 0.98, title_text, ha="center", va="top", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def analyze_line_hit_rates(df: pd.DataFrame, prop_analysis: pd.DataFrame, output_dir: Path = None):
    """
    Analyze hit rates for players at different line positions relative to their averages.

    Organized TEAM-BY-TEAM with separate tables for:
    - Points (PTS): Offsets of +/-5, +/-10
    - Assists (AST): Offsets of +/-2, +/-4
    - Rebounds (REB): Offsets of +/-2, +/-4

    Different offsets are used to account for scale differences:
    - Points have higher typical values (20-30), so +/-5 and +/-10 are appropriate
    - Assists/Rebounds have lower typical values (5-10), so +/-2 and +/-4 are more appropriate

    Hit Rate Definition: Percentage of games where the player's result was OVER the line.
    Example: If a player averages 20 PPG and went over 20 in 12 of 20 games, hit rate = 60%
    """
    logger.info("\n" + "=" * 100)
    logger.info("LINE HIT RATE ANALYSIS - TEAM-BY-TEAM")
    logger.info("=" * 100)
    logger.info("\n>>> Analyzing hit rates for Points, Assists, and Rebounds by team")
    logger.info(">>> Hit Rate = % of games player goes OVER the specified line")

    # Define stats to analyze - ONLY THE BIG 3
    # Each stat has different offsets based on typical scale:
    # - Points: +/-5, +/-10 (larger scale)
    # - Assists: +/-2, +/-4 (smaller scale)
    # - Rebounds: +/-2, +/-4 (smaller scale)
    stats_to_analyze = {
        "PTS": {
            "col": "PTS",
            "avg_col": "PPG",
            "name": "Points",
            "offsets": {"Avg-10": -10, "Avg-5": -5, "Avg+5": +5, "Avg+10": +10},
        },
        "AST": {
            "col": "AST",
            "avg_col": "APG",
            "name": "Assists",
            "offsets": {"Avg-4": -4, "Avg-2": -2, "Avg+2": +2, "Avg+4": +4},
        },
        "REB": {
            "col": "REB",
            "avg_col": "RPG",
            "name": "Rebounds",
            "offsets": {"Avg-4": -4, "Avg-2": -2, "Avg+2": +2, "Avg+4": +4},
        },
    }

    # Get team for each player
    player_teams = (
        df[["player_name", "player_team"]]
        .groupby("player_name")["player_team"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
    )
    player_teams.columns = ["player_name", "team"]

    # Merge team info
    prop_with_team = prop_analysis.merge(player_teams, on="player_name", how="left")

    # Filter for qualified players (≥10 games)
    qualified_players = prop_with_team[prop_with_team["games_played"] >= 10].copy()

    logger.info(f"\n>>> Analyzing {len(qualified_players)} qualified players (≥10 games)")

    # Calculate hit rates for ALL qualified players first
    all_results = []
    games_by_player = dict(tuple(df.groupby("player_name", sort=False)))

    for _, player_row in qualified_players.iterrows():
        player_name = player_row["player_name"]
        team = player_row["team"]

        # Get all games for this player
        player_games = games_by_player[player_name]

        if len(player_games) < 10:
            continue

        player_result = {"player_name": player_name, "team": team, "games": len(player_games)}

        # For each stat, store the average
        for stat_key, stat_info in stats_to_analyze.items():
            avg_value = player_row.get(stat_info["avg_col"], None)
            player_result[f"{stat_key}_avg"] = avg_value

            if pd.isna(avg_value) or avg_value == 0:
                # Skip if no average available
                for offset_name in stat_info["offsets"].keys():
                    player_result[f"{stat_key}_{offset_name}"] = np.nan
                continue

            # Calculate hit rates for each offset (stat-specific)
            for offset_name, offset_value in stat_info["offsets"].items():
                line = avg_value + offset_value

                # Count how many games player went OVER this line
                games_over = (player_games[stat_info["col"]] > line).sum()
                hit_rate = (games_over / len(player_games)) * 100

                player_result[f"{stat_key}_{offset_name}"] = hit_rate

        all_results.append(player_result)

    # Convert to DataFrame
    all_players_df = pd.DataFrame(all_results)

    # =========================================================================
    # TEAM-BY-TEAM ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("HIT RATE TABLES BY TEAM")
    logger.info("=" * 100)
    logger.info("\nAll values = % of games player went OVER the line")
    logger.info("For each team: 3 tables (Points, Assists, Rebounds)")

    # Get unique teams sorted
    teams = sorted(all_players_df["team"].dropna().unique())

    logger.info(f"\n>>> Analyzing {len(teams)} teams")

    # Store team results and figure paths
    results_by_team = {}
    saved_figures = []

    # Create output directory if provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # For each team, display 3 tables (PTS, AST, REB)
    for team in teams:
        team_players = all_players_df[all_players_df["team"] == team].copy()

        if len(team_players) == 0:
            continue

        results_by_team[team] = team_players

        logger.info("\n" + "=" * 100)
        logger.info(f"{team}")
        logger.info("=" * 100)

        # =====================================================================
        # TABLE 1: POINTS HIT RATES
        # =====================================================================
        logger.info("\n>>> POINTS HIT RATES (% of games OVER line)")

        # Sort by PPG descending, take top 15
        pts_cols = [
            "player_name",
            "PTS_avg",
            "games",
            "PTS_Avg-10",
            "PTS_Avg-5",
            "PTS_Avg+5",
            "PTS_Avg+10",
        ]
        pts_data = team_players[pts_cols].copy()
        pts_data = pts_data.sort_values("PTS_avg", ascending=False).head(15)
        pts_data.columns = ["Player", "Avg PPG", "Games", "Avg-10", "Avg-5", "Avg+5", "Avg+10"]

        # Round all numeric columns
        for col in ["Avg PPG", "Avg-10", "Avg-5", "Avg+5", "Avg+10"]:
            pts_data[col] = pts_data[col].round(1)

        print(
            "\n"
            + tabulate(pts_data, headers="keys", tablefmt="grid", showindex=False, floatfmt=".1f")
        )

        # Save as figure if output_dir provided
        if output_dir is not None:
            fig = create_table_figure(
                pts_data, f"{team}", "Points Hit Rates (% of games OVER line)"
            )
            filename = f"hit_rates_{team.replace(' ', '_')}_points.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            saved_figures.append(filepath)
            plt.close(fig)

        # =====================================================================
        # TABLE 2: ASSISTS HIT RATES
        # =====================================================================
        logger.info("\n>>> ASSISTS HIT RATES (% of games OVER line)")

        # Sort by APG descending, take top 15
        ast_cols = [
            "player_name",
            "AST_avg",
            "games",
            "AST_Avg-4",
            "AST_Avg-2",
            "AST_Avg+2",
            "AST_Avg+4",
        ]
        ast_data = team_players[ast_cols].copy()
        ast_data = ast_data.sort_values("AST_avg", ascending=False).head(15)
        ast_data.columns = ["Player", "Avg APG", "Games", "Avg-4", "Avg-2", "Avg+2", "Avg+4"]

        # Round all numeric columns
        for col in ["Avg APG", "Avg-4", "Avg-2", "Avg+2", "Avg+4"]:
            ast_data[col] = ast_data[col].round(1)

        print(
            "\n"
            + tabulate(ast_data, headers="keys", tablefmt="grid", showindex=False, floatfmt=".1f")
        )

        # Save as figure if output_dir provided
        if output_dir is not None:
            fig = create_table_figure(
                ast_data, f"{team}", "Assists Hit Rates (% of games OVER line)"
            )
            filename = f"hit_rates_{team.replace(' ', '_')}_assists.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            saved_figures.append(filepath)
            plt.close(fig)

        # =====================================================================
        # TABLE 3: REBOUNDS HIT RATES
        # =====================================================================
        logger.info("\n>>> REBOUNDS HIT RATES (% of games OVER line)")

        # Sort by RPG descending, take top 15
        reb_cols = [
            "player_name",
            "REB_avg",
            "games",
            "REB_Avg-4",
            "REB_Avg-2",
            "REB_Avg+2",
            "REB_Avg+4",
        ]
        reb_data = team_players[reb_cols].copy()
        reb_data = reb_data.sort_values("REB_avg", ascending=False).head(15)
        reb_data.columns = ["Player", "Avg RPG", "Games", "Avg-4", "Avg-2", "Avg+2", "Avg+4"]

        # Round all numeric columns
        for col in ["Avg RPG", "Avg-4", "Avg-2", "Avg+2", "Avg+4"]:
            reb_data[col] = reb_data[col].round(1)

        print(
            "\n"
            + tabulate(reb_data, headers="keys", tablefmt="grid", showindex=False, floatfmt=".1f")
        )

        # Save as figure if output_dir provided
        if output_dir is not None:
            fig = create_table_figure(
                reb_data, f"{team}", "Rebounds Hit Rates (% of games OVER line)"
            )
            filename = f"hit_rates_{team.replace(' ', '_')}_rebounds.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            saved_figures.append(filepath)
            plt.close(fig)

    # =========================================================================
    # LEAGUE-WIDE SUMMARY STATISTICS
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("LEAGUE-WIDE HIT RATE SUMMARY")
    logger.info("=" * 100)

    logger.info(f"\n>>> Average hit rates across all {len(all_players_df)} qualified players:")

    summary_data = []
    for stat_key, stat_info in stats_to_analyze.items():
        stat_summary = {"Stat": stat_info["name"]}
        for offset_name in stat_info["offsets"].keys():
            col_name = f"{stat_key}_{offset_name}"
            avg_hit_rate = all_players_df[col_name].mean()
            stat_summary[offset_name] = avg_hit_rate
        summary_data.append(stat_summary)

    summary_df = pd.DataFrame(summary_data)
    print(
        "\n"
        + tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".1f")
    )

    logger.info("\n>>> Interpretation Guide:")
    logger.info("    POINTS (scale: +/-5, +/-10):")
    logger.info("      - Avg-10: Very conservative line (should have high hit rate ~80-90%)")
    logger.info("      - Avg-5: Conservative line (should have hit rate ~60-70%)")
    logger.info("      - Avg+5: Aggressive line (should have hit rate ~30-40%)")
    logger.info("      - Avg+10: Very aggressive line (should have low hit rate ~10-20%)")
    logger.info("    ASSISTS & REBOUNDS (scale: +/-2, +/-4):")
    logger.info("      - Avg-4: Very conservative line (should have high hit rate ~80-90%)")
    logger.info("      - Avg-2: Conservative line (should have hit rate ~60-70%)")
    logger.info("      - Avg+2: Aggressive line (should have hit rate ~30-40%)")
    logger.info("      - Avg+4: Very aggressive line (should have low hit rate ~10-20%)")

    logger.info("\n>>> How to Use:")
    logger.info("    1. Points: Look for 70%+ hit rate at Avg-5 (reliable for accumulators)")
    logger.info(
        "    2. Assists/Rebounds: Look for 70%+ hit rate at Avg-2 (reliable for accumulators)"
    )
    logger.info("    3. Compare actual hit rates vs expected to find value")
    logger.info("    4. High aggressive rates (>40% at Avg+5/+2) indicate outperformance")
    logger.info("    5. Low hit rates at conservative lines suggest inconsistency")

    logger.info("\n✓ Line hit rate analysis complete!")

    # Log figure generation
    if output_dir is not None and len(saved_figures) > 0:
        logger.info(f"\n>>> Saved {len(saved_figures)} hit rate table figures to {output_dir}")
        logger.info(f"    ({len(teams)} teams × 3 tables each = {len(saved_figures)} figures)")

    # Return results for further analysis
    results = {
        "all_players": all_players_df,
        "by_team": results_by_team,
        "summary": summary_df,
        "figures": saved_figures,
    }

    return results
