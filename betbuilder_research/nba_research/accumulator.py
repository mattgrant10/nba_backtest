"""Preserved research calculations: accumulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.config import get_logger
from .plotting import dataset_title, show_figures

logger = get_logger(__name__)


def backtest_acca_across_dates(
    df: pd.DataFrame,
    prop_analysis: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    *,
    acca_template: dict | None = None,
):
    """
    Backtest accumulator bets across a date range.
    For each night, checks if the same acca structure would have won.
    Ignores players/teams that didn't play on that night.

    Args:
        df: Player game data
        prop_analysis: Player consistency metrics
        start_date: Start date (YYYY-MM-DD), defaults to earliest in data
        end_date: End date (YYYY-MM-DD), defaults to latest in data
    """
    logger.info("\n" + "=" * 100)
    logger.info("ACCUMULATOR BET BACKTESTING - DATE RANGE SIMULATION")
    logger.info("=" * 100)

    # Preserve the original specification; edit a copy of acca_template.json for extensions.
    if acca_template is None:
        from importlib.resources import files
        import json

        acca_template = json.loads(files("nba_research").joinpath("acca_template.json").read_text())

    # Convert dates
    df["game_date_only"] = pd.to_datetime(df["game_date_only"])

    if start_date is None:
        start_date = df["game_date_only"].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = df["game_date_only"].max()
    else:
        end_date = pd.to_datetime(end_date)

    logger.info(f"\n>>> Testing acca structure: {acca_template['name']}")
    logger.info(f">>> Date range: {start_date.date()} to {end_date.date()}")

    # Get all unique dates in range
    dates_in_range = df[(df["game_date_only"] >= start_date) & (df["game_date_only"] <= end_date)][
        "game_date_only"
    ].unique()
    dates_in_range = sorted(dates_in_range)

    logger.info(f">>> Total nights with games: {len(dates_in_range)}")

    # Results storage
    daily_results = []
    all_prop_results = []

    games_by_date = dict(tuple(df.groupby("game_date_only", sort=False)))

    # Test each night
    for test_date in dates_in_range:
        games_that_night = games_by_date[pd.Timestamp(test_date)]

        leg_results_for_night = []

        for leg_idx, leg in enumerate(acca_template["legs"]):
            props_evaluated = 0
            props_hit = 0
            props_in_leg = []

            for prop_template in leg["props"]:
                # Find player by last name
                player_games = games_that_night[
                    games_that_night["player_name"].str.contains(
                        prop_template["player_last_name"], case=False, na=False
                    )
                ]

                if len(player_games) == 0:
                    # Player didn't play this night, skip
                    continue

                # Get the player's performance that night
                player_game = player_games.iloc[0]

                # Get the stat value
                if prop_template["stat"] == "PTS":
                    result = player_game["PTS"]
                elif prop_template["stat"] == "AST":
                    result = player_game["AST"]
                elif prop_template["stat"] == "REB":
                    result = player_game["REB"]
                else:
                    continue

                # Check if prop hit
                if prop_template["over"]:
                    hit = result > prop_template["line"]
                else:
                    hit = result < prop_template["line"]

                margin = result - prop_template["line"]

                props_evaluated += 1
                if hit:
                    props_hit += 1

                # Get player consistency metrics
                player_stats = prop_analysis[
                    prop_analysis["player_name"].str.contains(
                        prop_template["player_last_name"], case=False, na=False
                    )
                ]

                if len(player_stats) > 0:
                    player_row = player_stats.iloc[0]
                    if prop_template["stat"] == "PTS":
                        cv = player_row.get("PTS_CV", np.nan)
                    elif prop_template["stat"] == "AST":
                        cv = player_row.get("AST_CV", np.nan)
                    elif prop_template["stat"] == "REB":
                        cv = player_row.get("REB_CV", np.nan)
                    else:
                        cv = np.nan
                else:
                    cv = np.nan

                props_in_leg.append(
                    {
                        "date": test_date,
                        "leg": leg_idx + 1,
                        "player": player_game["player_name"],
                        "stat": prop_template["stat"],
                        "line": prop_template["line"],
                        "result": result,
                        "margin": margin,
                        "hit": hit,
                        "cv": cv,
                    }
                )

            # Evaluate leg (all props must hit)
            if props_evaluated > 0:
                leg_hit = props_hit == props_evaluated
                hit_rate = (props_hit / props_evaluated) * 100
            else:
                leg_hit = False
                hit_rate = 0

            leg_results_for_night.append(
                {
                    "date": test_date,
                    "leg": leg_idx + 1,
                    "leg_name": leg["name"],
                    "props_evaluated": props_evaluated,
                    "props_hit": props_hit,
                    "hit_rate": hit_rate,
                    "leg_won": leg_hit,
                }
            )

            all_prop_results.extend(props_in_leg)

        # Check if entire acca won (all legs must win)
        if len(leg_results_for_night) > 0:
            legs_evaluated = len(leg_results_for_night)
            legs_won = sum(1 for leg in leg_results_for_night if leg["leg_won"])
            acca_won = all(leg["leg_won"] for leg in leg_results_for_night)

            daily_results.append(
                {
                    "date": test_date,
                    "legs_evaluated": legs_evaluated,
                    "legs_won": legs_won,
                    "acca_won": acca_won,
                }
            )

    # Convert to DataFrames
    daily_df = pd.DataFrame(daily_results)
    props_df = pd.DataFrame(all_prop_results)

    if len(daily_df) == 0:
        logger.warning("No games found in date range!")
        return None, None

    # Summary statistics
    logger.info("\n" + "=" * 100)
    logger.info("BACKTESTING RESULTS SUMMARY")
    logger.info("=" * 100)

    total_nights = len(daily_df)
    accas_won = daily_df["acca_won"].sum()
    win_rate = (accas_won / total_nights) * 100

    logger.info("\n>>> OVERALL PERFORMANCE:")
    logger.info(f"    Total Nights Tested: {total_nights}")
    logger.info(f"    Accas Won: {accas_won}")
    logger.info(f"    Accas Lost: {total_nights - accas_won}")
    logger.info(f"    Win Rate: {win_rate:.1f}%")

    # Display results by date
    logger.info("\n>>> RESULTS BY DATE:")
    display_daily = daily_df.copy()
    display_daily["date"] = display_daily["date"].dt.strftime("%Y-%m-%d")
    display_daily.columns = ["Date", "Legs Eval", "Legs Won", "Acca Won"]

    print("\n" + tabulate(display_daily.head(30), headers="keys", tablefmt="grid", showindex=False))
    if len(display_daily) > 30:
        print(f"\n... and {len(display_daily) - 30} more nights")
    print()

    # Winning streaks analysis
    logger.info("\n>>> STREAK ANALYSIS:")
    daily_df["win_streak"] = (
        daily_df["acca_won"]
        .astype(int)
        .groupby((daily_df["acca_won"] != daily_df["acca_won"].shift()).cumsum())
        .cumsum()
    )
    daily_df["loss_streak"] = (
        (~daily_df["acca_won"])
        .astype(int)
        .groupby((daily_df["acca_won"] != daily_df["acca_won"].shift()).cumsum())
        .cumsum()
    )

    max_win_streak = daily_df["win_streak"].max()
    max_loss_streak = daily_df["loss_streak"].max()

    logger.info(f"    Longest Win Streak: {max_win_streak} night(s)")
    logger.info(f"    Longest Loss Streak: {max_loss_streak} night(s)")

    # Individual prop success rates
    logger.info("\n>>> INDIVIDUAL PROP SUCCESS RATES:")
    prop_summary = (
        props_df.groupby(["player", "stat"])
        .agg({"hit": ["sum", "count", "mean"], "margin": "mean", "cv": "mean"})
        .round(2)
    )
    prop_summary.columns = ["Hits", "Total", "Hit Rate", "Avg Margin", "Avg CV"]
    prop_summary["Hit %"] = (prop_summary["Hit Rate"] * 100).round(1)
    prop_summary = prop_summary.sort_values("Hit %", ascending=False)

    print("\n" + tabulate(prop_summary.head(20), headers="keys", tablefmt="grid"))
    print()

    # VISUALIZATIONS
    logger.info("\n>>> Creating backtesting visualizations...")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Win rate over time
    ax1 = fig.add_subplot(gs[0, :])
    daily_df["date_str"] = daily_df["date"].dt.strftime("%m/%d")
    colors = ["green" if x else "red" for x in daily_df["acca_won"]]
    ax1.scatter(
        range(len(daily_df)),
        daily_df["acca_won"].astype(int),
        c=colors,
        s=100,
        alpha=0.6,
        edgecolor="black",
    )
    ax1.axhline(0.5, color="blue", linestyle="--", linewidth=2, label="50% baseline", alpha=0.5)
    ax1.set_xlabel("Night Index", fontweight="bold")
    ax1.set_ylabel("Won (1) / Lost (0)", fontweight="bold")
    ax1.set_title(
        dataset_title(f"Accumulator Results Over Time - Win Rate: {win_rate:.1f}%"),
        fontweight="bold",
        fontsize=14,
    )
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative wins
    ax2 = fig.add_subplot(gs[1, 0])
    daily_df["cumulative_wins"] = daily_df["acca_won"].cumsum()
    daily_df["cumulative_total"] = range(1, len(daily_df) + 1)
    ax2.plot(
        daily_df["cumulative_total"],
        daily_df["cumulative_wins"],
        linewidth=2,
        color="green",
        label="Wins",
    )
    ax2.plot(
        daily_df["cumulative_total"],
        daily_df["cumulative_total"] * (win_rate / 100),
        linewidth=2,
        linestyle="--",
        color="blue",
        label=f"Expected ({win_rate:.1f}%)",
    )
    ax2.fill_between(
        daily_df["cumulative_total"], 0, daily_df["cumulative_wins"], alpha=0.3, color="green"
    )
    ax2.set_xlabel("Nights Tested", fontweight="bold")
    ax2.set_ylabel("Cumulative Wins", fontweight="bold")
    ax2.set_title(dataset_title("Cumulative Wins Over Time"), fontweight="bold", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Win rate by day of week
    ax3 = fig.add_subplot(gs[1, 1])
    daily_df["day_of_week"] = daily_df["date"].dt.day_name()
    day_win_rate = daily_df.groupby("day_of_week")["acca_won"].mean() * 100
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_win_rate = day_win_rate.reindex([d for d in day_order if d in day_win_rate.index])
    ax3.bar(
        range(len(day_win_rate)),
        day_win_rate.values,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax3.set_xticks(range(len(day_win_rate)))
    ax3.set_xticklabels([d[:3] for d in day_win_rate.index], rotation=45)
    ax3.axhline(
        win_rate, color="red", linestyle="--", linewidth=2, label=f"Overall: {win_rate:.1f}%"
    )
    ax3.set_ylabel("Win Rate (%)", fontweight="bold")
    ax3.set_title(dataset_title("Win Rate by Day of Week"), fontweight="bold", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Legs won distribution
    ax4 = fig.add_subplot(gs[1, 2])
    legs_won_counts = daily_df["legs_won"].value_counts().sort_index()
    ax4.bar(
        legs_won_counts.index, legs_won_counts.values, color="orange", alpha=0.7, edgecolor="black"
    )
    ax4.set_xlabel("Number of Legs Won", fontweight="bold")
    ax4.set_ylabel("Frequency", fontweight="bold")
    ax4.set_title(
        dataset_title("Distribution of Legs Won per Night"), fontweight="bold", fontsize=12
    )
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Individual player success rates
    ax5 = fig.add_subplot(gs[2, :2])
    player_success = (
        props_df.groupby("player")["hit"]
        .agg(["sum", "count", "mean"])
        .sort_values("mean", ascending=True)
    )
    player_success["pct"] = player_success["mean"] * 100
    y_pos = range(len(player_success))
    colors_player = ["green" if x >= 0.5 else "red" for x in player_success["mean"]]
    ax5.barh(y_pos, player_success["pct"], color=colors_player, alpha=0.7, edgecolor="black")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([p.split()[-1] for p in player_success.index], fontsize=9)
    ax5.axvline(50, color="blue", linestyle="--", linewidth=2, label="50% baseline")
    ax5.set_xlabel("Hit Rate (%)", fontweight="bold")
    ax5.set_title(
        dataset_title("Individual Player Prop Success Rates"), fontweight="bold", fontsize=12
    )
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="x")

    # 6. Rolling win rate
    ax6 = fig.add_subplot(gs[2, 2])
    window = min(7, len(daily_df))
    daily_df["rolling_win_rate"] = (
        daily_df["acca_won"].rolling(window=window, min_periods=1).mean() * 100
    )
    ax6.plot(
        daily_df["rolling_win_rate"],
        linewidth=2,
        color="purple",
        label=f"{window}-night rolling avg",
    )
    ax6.axhline(
        win_rate, color="red", linestyle="--", linewidth=2, label=f"Overall: {win_rate:.1f}%"
    )
    ax6.fill_between(
        range(len(daily_df)),
        daily_df["rolling_win_rate"],
        win_rate,
        where=daily_df["rolling_win_rate"] >= win_rate,
        alpha=0.3,
        color="green",
    )
    ax6.fill_between(
        range(len(daily_df)),
        daily_df["rolling_win_rate"],
        win_rate,
        where=daily_df["rolling_win_rate"] < win_rate,
        alpha=0.3,
        color="red",
    )
    ax6.set_xlabel("Night Index", fontweight="bold")
    ax6.set_ylabel("Win Rate (%)", fontweight="bold")
    ax6.set_title(dataset_title(f"{window}-Night Rolling Win Rate"), fontweight="bold", fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Prop hit rate by stat type
    ax7 = fig.add_subplot(gs[3, 0])
    stat_success = props_df.groupby("stat")["hit"].mean() * 100
    ax7.bar(stat_success.index, stat_success.values, color="teal", alpha=0.7, edgecolor="black")
    ax7.axhline(50, color="red", linestyle="--", linewidth=2, label="50% baseline")
    ax7.set_ylabel("Hit Rate (%)", fontweight="bold")
    ax7.set_title(dataset_title("Success Rate by Stat Type"), fontweight="bold", fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")

    # 8. Margin distribution
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.hist(props_df["margin"], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax8.axvline(0, color="red", linestyle="--", linewidth=2, label="Break-even")
    ax8.axvline(
        props_df["margin"].mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {props_df['margin'].mean():.2f}",
    )
    ax8.set_xlabel("Margin (Result - Line)", fontweight="bold")
    ax8.set_ylabel("Frequency", fontweight="bold")
    ax8.set_title(
        dataset_title("Margin Distribution Across All Props"), fontweight="bold", fontsize=12
    )
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis="y")

    # 9. Consistency vs success
    ax9 = fig.add_subplot(gs[3, 2])
    valid_cv = props_df[props_df["cv"].notna()]
    colors_cv = ["green" if x else "red" for x in valid_cv["hit"]]
    ax9.scatter(valid_cv["cv"], valid_cv["margin"], c=colors_cv, alpha=0.6, s=80, edgecolor="black")
    ax9.axhline(0, color="blue", linestyle="--", linewidth=2, label="Break-even")
    ax9.axvline(0.40, color="orange", linestyle="--", linewidth=2, label="CV=0.40")
    ax9.set_xlabel("Player Consistency (CV)", fontweight="bold")
    ax9.set_ylabel("Margin", fontweight="bold")
    ax9.set_title(dataset_title("Player Consistency vs Margin"), fontweight="bold", fontsize=12)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    fig.suptitle(
        dataset_title(
            f"Accumulator Backtesting Analysis - {start_date.date()} to {end_date.date()}"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    show_figures()

    logger.info("\n✓ Date range backtesting complete!")

    return daily_df, props_df
