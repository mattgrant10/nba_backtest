#!/usr/bin/env python3
"""
Comprehensive NBA Player Statistics Analysis
Analyzes PlayerStatistics_2024_2025.csv with detailed logging and visualizations
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from src.config import setup_logging, get_logger
from src.visualization import (
    display_dataframe,
    display_summary_stats,
    plot_distributions,
    plot_correlation_matrix,
    plot_boxplots,
    plot_value_counts,
)

# Setup logging
setup_logging(level=20)  # INFO level
logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_and_display_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data and display comprehensive information."""
    logger.info("=" * 100)
    logger.info("STEP 1: LOADING RAW NBA PLAYER STATISTICS DATA")
    logger.info("=" * 100)

    logger.info(f"\n>>> Loading data from: {file_path}")
    logger.info(f">>> File size: {file_path.stat().st_size / 1024**2:.2f} MB")

    # Load data
    df = pd.read_csv(file_path, parse_dates=['gameDate'])

    logger.info(f"✓ Successfully loaded {len(df):,} rows × {len(df.columns)} columns")

    # Display raw data
    display_dataframe(df, "Raw Player Statistics Data", max_rows=20)

    # Show column types
    logger.info("\n>>> Column Data Types:")
    dtype_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values.astype(str),
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Unique': df.nunique().values
    })
    print("\n" + tabulate(dtype_df, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Memory usage
    logger.info(f"\n>>> Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 2: CLEANING AND PREPARING DATA")
    logger.info("=" * 100)

    df_clean = df.copy()

    # Create player full name
    logger.info("\n>>> Creating player full name...")
    df_clean['player_name'] = df_clean['firstName'] + ' ' + df_clean['lastName']

    # Convert date
    logger.info(">>> Processing dates...")
    df_clean['game_date'] = pd.to_datetime(df_clean['gameDate'])
    df_clean['game_date_only'] = df_clean['game_date'].dt.date
    df_clean['day_of_week'] = df_clean['game_date'].dt.day_name()
    df_clean['month'] = df_clean['game_date'].dt.month
    df_clean['week'] = df_clean['game_date'].dt.isocalendar().week

    # Create team names
    logger.info(">>> Creating team identifiers...")
    df_clean['player_team'] = df_clean['playerteamCity'] + ' ' + df_clean['playerteamName']
    df_clean['opponent_team'] = df_clean['opponentteamCity'] + ' ' + df_clean['opponentteamName']

    # Convert win to boolean
    df_clean['win'] = df_clean['win'].astype(bool)
    df_clean['home'] = df_clean['home'].astype(bool)

    # Create meaningful minutes played
    df_clean['minutes_played'] = df_clean['numMinutes']

    # Rename key stats for clarity
    stat_renames = {
        'points': 'PTS',
        'assists': 'AST',
        'reboundsTotal': 'REB',
        'threePointersMade': 'FG3M',
        'steals': 'STL',
        'blocks': 'BLK',
        'turnovers': 'TOV',
        'fieldGoalsMade': 'FGM',
        'fieldGoalsAttempted': 'FGA',
        'threePointersAttempted': 'FG3A',
        'freeThrowsMade': 'FTM',
        'freeThrowsAttempted': 'FTA',
        'plusMinusPoints': 'PLUS_MINUS'
    }

    for old, new in stat_renames.items():
        if old in df_clean.columns:
            df_clean[new] = df_clean[old]

    # Calculate shooting percentages (handle division by zero)
    logger.info(">>> Calculating shooting percentages...")
    df_clean['FG_PCT'] = np.where(df_clean['FGA'] > 0, df_clean['FGM'] / df_clean['FGA'], 0)
    df_clean['FG3_PCT'] = np.where(df_clean['FG3A'] > 0, df_clean['FG3M'] / df_clean['FG3A'], 0)
    df_clean['FT_PCT'] = np.where(df_clean['FTA'] > 0, df_clean['FTM'] / df_clean['FTA'], 0)

    # Calculate per-minute stats
    logger.info(">>> Calculating per-minute statistics...")
    df_clean['PTS_PER_MIN'] = np.where(df_clean['minutes_played'] > 0,
                                        df_clean['PTS'] / df_clean['minutes_played'], 0)
    df_clean['AST_PER_MIN'] = np.where(df_clean['minutes_played'] > 0,
                                        df_clean['AST'] / df_clean['minutes_played'], 0)
    df_clean['REB_PER_MIN'] = np.where(df_clean['minutes_played'] > 0,
                                        df_clean['REB'] / df_clean['minutes_played'], 0)

    # Filter out players with < 5 minutes (likely DNP or garbage time)
    logger.info("\n>>> Filtering players with < 5 minutes played...")
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['minutes_played'] >= 5].copy()
    filtered_count = initial_count - len(df_clean)
    logger.info(f"    Removed {filtered_count:,} rows ({filtered_count/initial_count*100:.1f}%)")
    logger.info(f"    Remaining: {len(df_clean):,} rows")

    display_dataframe(df_clean, "Cleaned Player Statistics Data", max_rows=15)

    return df_clean


def analyze_player_level_stats(df: pd.DataFrame):
    """Detailed player-level analysis."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 3: PLAYER-LEVEL ANALYSIS")
    logger.info("=" * 100)

    # Key stats to analyze
    key_stats = ['PTS', 'AST', 'REB', 'FG3M', 'STL', 'BLK', 'TOV']

    # Overall statistics
    logger.info("\n>>> Overall Player Statistics Summary:")
    display_summary_stats(df, columns=key_stats, title="Key Statistics Summary")

    # Distribution plots
    logger.info("\n>>> Plotting distributions of key statistics...")
    plot_distributions(df, columns=key_stats, bins=40, show=True)

    # Shooting percentages
    logger.info("\n>>> Shooting Percentages Summary:")
    shooting_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
    display_summary_stats(df, columns=shooting_stats, title="Shooting Percentages")
    plot_distributions(df, columns=shooting_stats, bins=30, show=True)

    # Per-minute stats
    logger.info("\n>>> Per-Minute Statistics Summary:")
    per_min_stats = ['PTS_PER_MIN', 'AST_PER_MIN', 'REB_PER_MIN']
    display_summary_stats(df, columns=per_min_stats, title="Per-Minute Statistics")

    # Top performers (by game)
    logger.info("\n>>> Top 20 Single-Game Performances (Points):")
    top_scorers = df.nlargest(20, 'PTS')[
        ['player_name', 'game_date_only', 'player_team', 'opponent_team',
         'PTS', 'AST', 'REB', 'FG3M', 'minutes_played', 'win', 'home']
    ]
    print("\n" + tabulate(top_scorers, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Correlation matrix
    logger.info("\n>>> Plotting correlation matrix of key statistics...")
    plot_correlation_matrix(df, columns=key_stats + ['minutes_played'], method='spearman', show=True)

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
    player_agg = df.groupby('player_name').agg({
        'personId': 'first',
        'gameId': 'count',  # Games played
        'PTS': 'mean',
        'AST': 'mean',
        'REB': 'mean',
        'FG3M': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TOV': 'mean',
        'minutes_played': 'mean',
        'FG_PCT': 'mean',
        'FG3_PCT': 'mean',
        'FT_PCT': 'mean',
        'win': 'mean',  # Win percentage
        'home': 'mean',  # Home game percentage
        'PLUS_MINUS': 'mean'
    }).round(2)

    # Rename columns
    player_agg.columns = [
        'player_id', 'games_played', 'PPG', 'APG', 'RPG', 'FG3M_PG',
        'SPG', 'BPG', 'TPG', 'MPG', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'WIN_PCT', 'HOME_PCT', 'AVG_PLUS_MINUS'
    ]

    # Reset index
    player_agg = player_agg.reset_index()

    # Filter players with at least 10 games
    logger.info("\n>>> Filtering players with at least 10 games played...")
    initial_players = len(player_agg)
    player_agg = player_agg[player_agg['games_played'] >= 10].copy()
    logger.info(f"    Players with ≥10 games: {len(player_agg):,} / {initial_players:,}")

    # Calculate additional metrics
    logger.info("\n>>> Calculating composite metrics...")
    player_agg['PTS_AST'] = player_agg['PPG'] + player_agg['APG']
    player_agg['PTS_REB'] = player_agg['PPG'] + player_agg['RPG']
    player_agg['PTS_AST_REB'] = player_agg['PPG'] + player_agg['APG'] + player_agg['RPG']

    # Sort by PPG
    player_agg = player_agg.sort_values('PPG', ascending=False)

    display_dataframe(player_agg, "Player Season Averages", max_rows=25)
    display_summary_stats(player_agg, title="Player Season Averages - Summary Statistics")

    # Top players
    logger.info("\n>>> Top 20 Players by Points Per Game:")
    top_ppg = player_agg.nlargest(20, 'PPG')[
        ['player_name', 'games_played', 'PPG', 'APG', 'RPG', 'FG3M_PG',
         'FG_PCT', 'MPG', 'WIN_PCT']
    ]
    print("\n" + tabulate(top_ppg, headers='keys', tablefmt='grid', showindex=False))
    print()

    logger.info("\n>>> Top 20 Players by Assists Per Game:")
    top_apg = player_agg.nlargest(20, 'APG')[
        ['player_name', 'games_played', 'APG', 'PPG', 'RPG', 'MPG', 'WIN_PCT']
    ]
    print("\n" + tabulate(top_apg, headers='keys', tablefmt='grid', showindex=False))
    print()

    logger.info("\n>>> Top 20 Players by Rebounds Per Game:")
    top_rpg = player_agg.nlargest(20, 'RPG')[
        ['player_name', 'games_played', 'RPG', 'PPG', 'APG', 'MPG', 'WIN_PCT']
    ]
    print("\n" + tabulate(top_rpg, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Plot distributions of averages
    logger.info("\n>>> Plotting distributions of player averages...")
    plot_distributions(player_agg, columns=['PPG', 'APG', 'RPG', 'FG3M_PG'], bins=30, show=True)

    # Shooting efficiency
    logger.info("\n>>> Top 20 Most Efficient Shooters (FG% with ≥10 PPG):")
    efficient_shooters = player_agg[player_agg['PPG'] >= 10].nlargest(20, 'FG_PCT')[
        ['player_name', 'games_played', 'PPG', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    ]
    print("\n" + tabulate(efficient_shooters, headers='keys', tablefmt='grid', showindex=False))
    print()

    return player_agg


def aggregate_by_team(df: pd.DataFrame):
    """Aggregate statistics by team."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 5: TEAM-LEVEL AGGREGATION")
    logger.info("=" * 100)

    logger.info("\n>>> Aggregating statistics by team...")

    team_agg = df.groupby('player_team').agg({
        'gameId': 'count',
        'PTS': 'mean',
        'AST': 'mean',
        'REB': 'mean',
        'FG3M': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TOV': 'mean',
        'FG_PCT': 'mean',
        'FG3_PCT': 'mean',
        'win': 'mean',
        'PLUS_MINUS': 'mean',
        'personId': 'nunique'  # Unique players
    }).round(2)

    team_agg.columns = [
        'total_player_games', 'avg_PTS', 'avg_AST', 'avg_REB', 'avg_3PM',
        'avg_STL', 'avg_BLK', 'avg_TOV', 'avg_FG_PCT', 'avg_3P_PCT',
        'win_rate', 'avg_plus_minus', 'unique_players'
    ]

    team_agg = team_agg.sort_values('win_rate', ascending=False).reset_index()

    display_dataframe(team_agg, "Team Statistics", max_rows=30)

    logger.info("\n>>> All Teams Ranked by Win Rate:")
    print("\n" + tabulate(team_agg, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Plot team statistics
    logger.info("\n>>> Plotting team statistics distributions...")
    plot_distributions(team_agg, columns=['avg_PTS', 'avg_AST', 'avg_REB', 'win_rate'], bins=15, show=True)

    return team_agg


def analyze_home_away_performance(df: pd.DataFrame):
    """Analyze home vs away performance."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 6: HOME VS AWAY PERFORMANCE ANALYSIS")
    logger.info("=" * 100)

    logger.info("\n>>> Comparing home vs away statistics...")

    home_away_stats = df.groupby('home').agg({
        'gameId': 'count',
        'PTS': 'mean',
        'AST': 'mean',
        'REB': 'mean',
        'FG_PCT': 'mean',
        'FG3_PCT': 'mean',
        'win': 'mean',
        'PLUS_MINUS': 'mean'
    }).round(3)

    home_away_stats.index = ['Away', 'Home']
    home_away_stats.columns = [
        'Games', 'Avg PTS', 'Avg AST', 'Avg REB',
        'FG%', '3P%', 'Win Rate', 'Avg +/-'
    ]

    logger.info("\n>>> Home vs Away Comparison:")
    print("\n" + tabulate(home_away_stats, headers='keys', tablefmt='grid'))
    print()

    # Calculate differences
    if len(home_away_stats) == 2:
        diff = home_away_stats.loc['Home'] - home_away_stats.loc['Away']
        logger.info("\n>>> Home Advantage (Home - Away):")
        print("\n" + tabulate(diff.to_frame('Difference'), headers='keys', tablefmt='grid'))
        print()

    # Visualize
    logger.info("\n>>> Plotting home vs away distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (stat, ax) in enumerate(zip(['PTS', 'AST', 'REB', 'FG_PCT'], axes.flatten())):
        df.boxplot(column=stat, by='home', ax=ax)
        ax.set_title(f'{stat} - Home vs Away')
        ax.set_xlabel('Home (0=Away, 1=Home)')
        ax.set_ylabel(stat)

    plt.tight_layout()
    plt.show()

    return home_away_stats


def analyze_win_loss_performance(df: pd.DataFrame):
    """Analyze performance in wins vs losses."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 7: WIN VS LOSS PERFORMANCE ANALYSIS")
    logger.info("=" * 100)

    logger.info("\n>>> Comparing statistics in wins vs losses...")

    win_loss_stats = df.groupby('win').agg({
        'gameId': 'count',
        'PTS': 'mean',
        'AST': 'mean',
        'REB': 'mean',
        'FG_PCT': 'mean',
        'FG3_PCT': 'mean',
        'TOV': 'mean',
        'PLUS_MINUS': 'mean',
        'STL': 'mean',
        'BLK': 'mean'
    }).round(3)

    win_loss_stats.index = ['Loss', 'Win']
    win_loss_stats.columns = [
        'Games', 'Avg PTS', 'Avg AST', 'Avg REB',
        'FG%', '3P%', 'Avg TOV', 'Avg +/-', 'Avg STL', 'Avg BLK'
    ]

    logger.info("\n>>> Win vs Loss Comparison:")
    print("\n" + tabulate(win_loss_stats, headers='keys', tablefmt='grid'))
    print()

    # Calculate differences
    if len(win_loss_stats) == 2:
        diff = win_loss_stats.loc['Win'] - win_loss_stats.loc['Loss']
        logger.info("\n>>> Win Impact (Win - Loss):")
        print("\n" + tabulate(diff.to_frame('Difference'), headers='keys', tablefmt='grid'))
        print()

    return win_loss_stats


def analyze_temporal_trends(df: pd.DataFrame):
    """Analyze trends over time."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 8: TEMPORAL TRENDS ANALYSIS")
    logger.info("=" * 100)

    # By day of week
    logger.info("\n>>> Performance by Day of Week:")
    day_stats = df.groupby('day_of_week').agg({
        'gameId': 'count',
        'PTS': 'mean',
        'AST': 'mean',
        'REB': 'mean',
        'win': 'mean'
    }).round(2)

    day_stats.columns = ['Games', 'Avg PTS', 'Avg AST', 'Avg REB', 'Win Rate']

    # Reorder by day
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = day_stats.reindex([d for d in day_order if d in day_stats.index])

    print("\n" + tabulate(day_stats, headers='keys', tablefmt='grid'))
    print()

    # By date (time series)
    logger.info("\n>>> Daily aggregated statistics over time...")
    daily_stats = df.groupby('game_date_only').agg({
        'gameId': 'count',
        'PTS': 'mean',
        'AST': 'mean',
        'REB': 'mean',
        'win': 'mean'
    }).reset_index()

    daily_stats.columns = ['Date', 'Games', 'Avg PTS', 'Avg AST', 'Avg REB', 'Win Rate']
    display_dataframe(daily_stats, "Daily Statistics Over Time", max_rows=20)

    # Plot time series
    logger.info("\n>>> Plotting temporal trends...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(daily_stats['Date'], daily_stats['Avg PTS'], marker='o', markersize=3)
    axes[0, 0].set_title('Average Points Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Points')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    axes[0, 1].plot(daily_stats['Date'], daily_stats['Avg AST'], marker='o', markersize=3, color='orange')
    axes[0, 1].set_title('Average Assists Over Time')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Assists')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 0].plot(daily_stats['Date'], daily_stats['Avg REB'], marker='o', markersize=3, color='green')
    axes[1, 0].set_title('Average Rebounds Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Rebounds')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    axes[1, 1].plot(daily_stats['Date'], daily_stats['Win Rate'], marker='o', markersize=3, color='red')
    axes[1, 1].set_title('Win Rate Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return daily_stats


def identify_prop_bet_opportunities(df: pd.DataFrame, player_agg: pd.DataFrame):
    """Identify potential prop bet opportunities based on consistency."""
    logger.info("\n" + "=" * 100)
    logger.info("STEP 9: PROP BET OPPORTUNITY IDENTIFICATION")
    logger.info("=" * 100)

    logger.info("\n>>> Analyzing player consistency for prop betting...")

    # Calculate std deviation for each player
    player_std = df.groupby('player_name').agg({
        'PTS': 'std',
        'AST': 'std',
        'REB': 'std',
        'FG3M': 'std',
        'gameId': 'count'
    }).round(2)

    player_std.columns = ['PTS_STD', 'AST_STD', 'REB_STD', 'FG3M_STD', 'games']

    # Merge with averages
    prop_analysis = player_agg.merge(player_std, left_on='player_name', right_index=True, how='inner')

    # Calculate coefficient of variation (lower = more consistent)
    prop_analysis['PTS_CV'] = (prop_analysis['PTS_STD'] / prop_analysis['PPG']).round(3)
    prop_analysis['AST_CV'] = (prop_analysis['AST_STD'] / prop_analysis['APG']).replace([np.inf, -np.inf], np.nan).round(3)
    prop_analysis['REB_CV'] = (prop_analysis['REB_STD'] / prop_analysis['RPG']).round(3)
    prop_analysis['FG3M_CV'] = (prop_analysis['FG3M_STD'] / prop_analysis['FG3M_PG']).replace([np.inf, -np.inf], np.nan).round(3)

    # Filter for relevant players (≥20 games, ≥15 PPG)
    prop_candidates = prop_analysis[
        (prop_analysis['games_played'] >= 20) &
        (prop_analysis['PPG'] >= 15)
    ].copy()

    logger.info(f"\n>>> Found {len(prop_candidates)} players meeting criteria (≥20 games, ≥15 PPG)")

    # Most consistent scorers (low CV)
    logger.info("\n>>> Top 15 Most Consistent Scorers (Low Points CV):")
    consistent_scorers = prop_candidates.nlargest(15, 'PPG').nsmallest(15, 'PTS_CV')[
        ['player_name', 'games_played', 'PPG', 'PTS_STD', 'PTS_CV', 'WIN_PCT']
    ]
    print("\n" + tabulate(consistent_scorers, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Most consistent assist players
    logger.info("\n>>> Top 15 Most Consistent Assist Players (Low Assists CV, ≥5 APG):")
    consistent_assists = prop_candidates[prop_candidates['APG'] >= 5].nsmallest(15, 'AST_CV')[
        ['player_name', 'games_played', 'APG', 'AST_STD', 'AST_CV', 'WIN_PCT']
    ]
    print("\n" + tabulate(consistent_assists, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Most consistent rebounders
    logger.info("\n>>> Top 15 Most Consistent Rebounders (Low Rebounds CV, ≥8 RPG):")
    consistent_rebounds = prop_candidates[prop_candidates['RPG'] >= 8].nsmallest(15, 'REB_CV')[
        ['player_name', 'games_played', 'RPG', 'REB_STD', 'REB_CV', 'WIN_PCT']
    ]
    print("\n" + tabulate(consistent_rebounds, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Most consistent three-point shooters
    logger.info("\n>>> Top 15 Most Consistent Three-Point Shooters (Low 3PM CV, ≥2 3PM/G):")
    consistent_threes = prop_candidates[prop_candidates['FG3M_PG'] >= 2].nsmallest(15, 'FG3M_CV')[
        ['player_name', 'games_played', 'FG3M_PG', 'FG3M_STD', 'FG3M_CV', 'WIN_PCT']
    ]
    print("\n" + tabulate(consistent_threes, headers='keys', tablefmt='grid', showindex=False))
    print()

    # Scatter plots: Average vs Consistency (now including 3-pointers)
    logger.info("\n>>> Plotting consistency analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Points
    axes[0, 0].scatter(prop_candidates['PPG'], prop_candidates['PTS_CV'], alpha=0.6, s=80)
    axes[0, 0].set_xlabel('Points Per Game')
    axes[0, 0].set_ylabel('Coefficient of Variation (lower = more consistent)')
    axes[0, 0].set_title('Scoring Consistency vs Average')
    axes[0, 0].grid(True, alpha=0.3)

    # Assists
    axes[0, 1].scatter(prop_candidates['APG'], prop_candidates['AST_CV'], alpha=0.6, s=80, color='orange')
    axes[0, 1].set_xlabel('Assists Per Game')
    axes[0, 1].set_ylabel('Coefficient of Variation (lower = more consistent)')
    axes[0, 1].set_title('Assist Consistency vs Average')
    axes[0, 1].grid(True, alpha=0.3)

    # Rebounds
    axes[1, 0].scatter(prop_candidates['RPG'], prop_candidates['REB_CV'], alpha=0.6, s=80, color='green')
    axes[1, 0].set_xlabel('Rebounds Per Game')
    axes[1, 0].set_ylabel('Coefficient of Variation (lower = more consistent)')
    axes[1, 0].set_title('Rebound Consistency vs Average')
    axes[1, 0].grid(True, alpha=0.3)

    # Three-Pointers
    axes[1, 1].scatter(prop_candidates['FG3M_PG'], prop_candidates['FG3M_CV'], alpha=0.6, s=80, color='red')
    axes[1, 1].set_xlabel('Three-Pointers Per Game')
    axes[1, 1].set_ylabel('Coefficient of Variation (lower = more consistent)')
    axes[1, 1].set_title('Three-Point Consistency vs Average')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Team-by-team consistency analysis
    logger.info("\n" + "=" * 100)
    logger.info("TEAM-BY-TEAM CONSISTENCY ANALYSIS")
    logger.info("=" * 100)
    logger.info("\n>>> Creating team-by-team scatterplots for consistency vs performance...")

    # Get team for each player (most common team they played for)
    player_teams = df[['player_name', 'player_team']].groupby('player_name')['player_team'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()
    player_teams.columns = ['player_name', 'team']

    # Merge team info into prop_analysis
    prop_with_team = prop_analysis.merge(player_teams, on='player_name', how='left')

    # Get all unique teams
    teams = sorted(prop_with_team['team'].dropna().unique())
    logger.info(f">>> Analyzing {len(teams)} teams")

    # Create team-by-team scatterplots for POINTS
    logger.info("\n>>> Creating Points Consistency by Team plots...")
    n_teams = len(teams)
    n_cols = 5
    n_rows = (n_teams + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, team in enumerate(teams):
        team_data = prop_with_team[
            (prop_with_team['team'] == team) &
            (prop_with_team['games_played'] >= 10)
        ]

        if len(team_data) > 0:
            axes[idx].scatter(team_data['PPG'], team_data['PTS_CV'], alpha=0.7, s=100)

            # Add player labels for top scorers
            top_scorers = team_data.nlargest(3, 'PPG')
            for _, player in top_scorers.iterrows():
                axes[idx].annotate(
                    player['player_name'].split()[-1],  # Last name only
                    (player['PPG'], player['PTS_CV']),
                    fontsize=8,
                    alpha=0.7
                )

            axes[idx].set_xlabel('Points Per Game', fontsize=9)
            axes[idx].set_ylabel('CV (lower = consistent)', fontsize=9)
            axes[idx].set_title(f'{team}', fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_teams, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Points Consistency vs PPG by Team', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

    # Create team-by-team scatterplots for ASSISTS
    logger.info("\n>>> Creating Assists Consistency by Team plots...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, team in enumerate(teams):
        team_data = prop_with_team[
            (prop_with_team['team'] == team) &
            (prop_with_team['games_played'] >= 10) &
            (prop_with_team['APG'] >= 2)  # Filter for relevant assist players
        ]

        if len(team_data) > 0:
            axes[idx].scatter(team_data['APG'], team_data['AST_CV'], alpha=0.7, s=100, color='orange')

            # Add player labels for top assist players
            top_assist = team_data.nlargest(3, 'APG')
            for _, player in top_assist.iterrows():
                axes[idx].annotate(
                    player['player_name'].split()[-1],  # Last name only
                    (player['APG'], player['AST_CV']),
                    fontsize=8,
                    alpha=0.7
                )

            axes[idx].set_xlabel('Assists Per Game', fontsize=9)
            axes[idx].set_ylabel('CV (lower = consistent)', fontsize=9)
            axes[idx].set_title(f'{team}', fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_teams, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Assists Consistency vs APG by Team', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

    # Create team-by-team scatterplots for REBOUNDS
    logger.info("\n>>> Creating Rebounds Consistency by Team plots...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, team in enumerate(teams):
        team_data = prop_with_team[
            (prop_with_team['team'] == team) &
            (prop_with_team['games_played'] >= 10) &
            (prop_with_team['RPG'] >= 3)  # Filter for relevant rebounders
        ]

        if len(team_data) > 0:
            axes[idx].scatter(team_data['RPG'], team_data['REB_CV'], alpha=0.7, s=100, color='green')

            # Add player labels for top rebounders
            top_reb = team_data.nlargest(3, 'RPG')
            for _, player in top_reb.iterrows():
                axes[idx].annotate(
                    player['player_name'].split()[-1],  # Last name only
                    (player['RPG'], player['REB_CV']),
                    fontsize=8,
                    alpha=0.7
                )

            axes[idx].set_xlabel('Rebounds Per Game', fontsize=9)
            axes[idx].set_ylabel('CV (lower = consistent)', fontsize=9)
            axes[idx].set_title(f'{team}', fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_teams, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Rebounds Consistency vs RPG by Team', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

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
        print(f"PAGE {page_num + 1} of {total_pages} - Teams {start_idx + 1}-{end_idx} of {len(teams)}")
        print("=" * 100)

        for team in teams_on_page:
            team_data = prop_with_team[
                (prop_with_team['team'] == team) &
                (prop_with_team['games_played'] >= 10)
            ].copy()

            if len(team_data) > 0:
                # Select relevant columns and sort by PPG
                consistency_table = team_data[[
                    'player_name',
                    'games_played',
                    'PPG', 'PTS_CV',
                    'APG', 'AST_CV',
                    'RPG', 'REB_CV',
                    'FG3M_PG', 'FG3M_CV'
                ]].sort_values('PPG', ascending=False).round(2)

                # Rename columns for clarity
                consistency_table.columns = [
                    'Player',
                    'Games',
                    'PPG', 'PTS_CV',
                    'APG', 'AST_CV',
                    'RPG', 'REB_CV',
                    '3PM', '3PM_CV'
                ]

                logger.info("\n" + "-" * 100)
                logger.info(f"{team} - CONSISTENCY METRICS")
                logger.info("-" * 100)
                logger.info(f"Players: {len(team_data)} | Games played: ≥10")
                logger.info("CV = Coefficient of Variation (Lower = More Consistent)")

                print("\n" + tabulate(
                    consistency_table,
                    headers='keys',
                    tablefmt='grid',
                    showindex=False,
                    floatfmt='.2f'
                ))
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
        (prop_analysis['games_played'] >= 15) &
        (prop_analysis['PTS_CV'] < 0.40) &
        (prop_analysis['AST_CV'] < 0.40) &
        (prop_analysis['REB_CV'] < 0.40) &
        (prop_analysis['FG3M_CV'] < 0.40)
    ].copy()

    logger.info(f"\n>>> Found {len(high_consistency)} players with CV < 0.40 across ALL categories (PTS, AST, REB, 3PM)")

    if len(high_consistency) > 0:
        logger.info("\n>>> ELITE CONSISTENCY - All Stats CV < 0.40:")
        elite_table = high_consistency[[
            'player_name',
            'games_played',
            'PPG', 'PTS_CV',
            'APG', 'AST_CV',
            'RPG', 'REB_CV',
            'FG3M_PG', 'FG3M_CV',
            'WIN_PCT'
        ]].sort_values('PPG', ascending=False).round(2)

        elite_table.columns = [
            'Player', 'Games',
            'PPG', 'PTS_CV',
            'APG', 'AST_CV',
            'RPG', 'REB_CV',
            '3PM', '3PM_CV',
            'Win%'
        ]

        print("\n" + tabulate(elite_table, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
        print()
    else:
        logger.info("\n>>> No players found with CV < 0.40 across ALL categories")
        logger.info(">>> Relaxing criteria...")

    # Also show players with CV < 0.40 for Points only
    logger.info("\n>>> POINTS CONSISTENCY - PTS_CV < 0.40:")
    pts_consistent = prop_analysis[
        (prop_analysis['games_played'] >= 15) &
        (prop_analysis['PPG'] >= 10) &
        (prop_analysis['PTS_CV'] < 0.40)
    ].sort_values('PPG', ascending=False)[
        ['player_name', 'games_played', 'PPG', 'PTS_STD', 'PTS_CV', 'WIN_PCT']
    ].head(20).round(2)

    pts_consistent.columns = ['Player', 'Games', 'PPG', 'PTS_STD', 'PTS_CV', 'Win%']
    print("\n" + tabulate(pts_consistent, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    print()

    # Assists consistency
    logger.info("\n>>> ASSISTS CONSISTENCY - AST_CV < 0.40:")
    ast_consistent = prop_analysis[
        (prop_analysis['games_played'] >= 15) &
        (prop_analysis['APG'] >= 3) &
        (prop_analysis['AST_CV'] < 0.40)
    ].sort_values('APG', ascending=False)[
        ['player_name', 'games_played', 'APG', 'AST_STD', 'AST_CV', 'WIN_PCT']
    ].head(20).round(2)

    ast_consistent.columns = ['Player', 'Games', 'APG', 'AST_STD', 'AST_CV', 'Win%']
    print("\n" + tabulate(ast_consistent, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    print()

    # Rebounds consistency
    logger.info("\n>>> REBOUNDS CONSISTENCY - REB_CV < 0.40:")
    reb_consistent = prop_analysis[
        (prop_analysis['games_played'] >= 15) &
        (prop_analysis['RPG'] >= 5) &
        (prop_analysis['REB_CV'] < 0.40)
    ].sort_values('RPG', ascending=False)[
        ['player_name', 'games_played', 'RPG', 'REB_STD', 'REB_CV', 'WIN_PCT']
    ].head(20).round(2)

    reb_consistent.columns = ['Player', 'Games', 'RPG', 'REB_STD', 'REB_CV', 'Win%']
    print("\n" + tabulate(reb_consistent, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    print()

    # Three-pointers consistency
    logger.info("\n>>> THREE-POINTERS CONSISTENCY - FG3M_CV < 0.40:")
    fg3_consistent = prop_analysis[
        (prop_analysis['games_played'] >= 15) &
        (prop_analysis['FG3M_PG'] >= 2) &
        (prop_analysis['FG3M_CV'] < 0.40)
    ].sort_values('FG3M_PG', ascending=False)[
        ['player_name', 'games_played', 'FG3M_PG', 'FG3M_STD', 'FG3M_CV', 'WIN_PCT']
    ].head(20).round(2)

    fg3_consistent.columns = ['Player', 'Games', '3PM', '3PM_STD', '3PM_CV', 'Win%']
    print("\n" + tabulate(fg3_consistent, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    print()

    logger.info("\n✓ Simulation analysis complete - prop bet targets identified!")

    return prop_analysis


def analyze_line_hit_rates(df: pd.DataFrame, prop_analysis: pd.DataFrame):
    """
    Analyze hit rates for players at different line positions relative to their averages.

    Organized TEAM-BY-TEAM with separate tables for:
    - Points (PTS)
    - Assists (AST)
    - Rebounds (REB)

    For each team, shows all qualified players with hit rates at:
    - Average - 10
    - Average - 5
    - Average + 5
    - Average + 10

    Hit Rate Definition: Percentage of games where the player's result was OVER the line.
    Example: If a player averages 20 PPG and went over 20 in 12 of 20 games, hit rate = 60%
    """
    logger.info("\n" + "=" * 100)
    logger.info("LINE HIT RATE ANALYSIS - TEAM-BY-TEAM")
    logger.info("=" * 100)
    logger.info("\n>>> Analyzing hit rates for Points, Assists, and Rebounds by team")
    logger.info(">>> Hit Rate = % of games player goes OVER the specified line")

    # Define stats to analyze - ONLY THE BIG 3
    stats_to_analyze = {
        'PTS': {'col': 'PTS', 'avg_col': 'PPG', 'name': 'Points'},
        'AST': {'col': 'AST', 'avg_col': 'APG', 'name': 'Assists'},
        'REB': {'col': 'REB', 'avg_col': 'RPG', 'name': 'Rebounds'}
    }

    # Line offsets to test - ONLY +/-5 and +/-10
    offsets = {
        'Avg-10': -10,
        'Avg-5': -5,
        'Avg+5': +5,
        'Avg+10': +10
    }

    # Get team for each player
    player_teams = df[['player_name', 'player_team']].groupby('player_name')['player_team'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()
    player_teams.columns = ['player_name', 'team']

    # Merge team info
    prop_with_team = prop_analysis.merge(player_teams, on='player_name', how='left')

    # Filter for qualified players (≥10 games)
    qualified_players = prop_with_team[prop_with_team['games_played'] >= 10].copy()

    logger.info(f"\n>>> Analyzing {len(qualified_players)} qualified players (≥10 games)")

    # Calculate hit rates for ALL qualified players first
    all_results = []

    for _, player_row in qualified_players.iterrows():
        player_name = player_row['player_name']
        team = player_row['team']

        # Get all games for this player
        player_games = df[df['player_name'] == player_name]

        if len(player_games) < 10:
            continue

        player_result = {
            'player_name': player_name,
            'team': team,
            'games': len(player_games)
        }

        # For each stat, store the average
        for stat_key, stat_info in stats_to_analyze.items():
            avg_value = player_row.get(stat_info['avg_col'], None)
            player_result[f'{stat_key}_avg'] = avg_value

            if pd.isna(avg_value) or avg_value == 0:
                # Skip if no average available
                for offset_name in offsets.keys():
                    player_result[f'{stat_key}_{offset_name}'] = np.nan
                continue

            # Calculate hit rates for each offset
            for offset_name, offset_value in offsets.items():
                line = avg_value + offset_value

                # Count how many games player went OVER this line
                games_over = (player_games[stat_info['col']] > line).sum()
                hit_rate = (games_over / len(player_games)) * 100

                player_result[f'{stat_key}_{offset_name}'] = hit_rate

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
    teams = sorted(all_players_df['team'].dropna().unique())

    logger.info(f"\n>>> Analyzing {len(teams)} teams")

    # Store team results for return
    results_by_team = {}

    # For each team, display 3 tables (PTS, AST, REB)
    for team in teams:
        team_players = all_players_df[all_players_df['team'] == team].copy()

        if len(team_players) == 0:
            continue

        results_by_team[team] = team_players

        logger.info("\n" + "=" * 100)
        logger.info(f"{team}")
        logger.info("=" * 100)

        # =====================================================================
        # TABLE 1: POINTS HIT RATES
        # =====================================================================
        logger.info(f"\n>>> POINTS HIT RATES (% of games OVER line)")

        # Sort by PPG descending, take top 15
        pts_cols = ['player_name', 'PTS_avg', 'games', 'PTS_Avg-10', 'PTS_Avg-5', 'PTS_Avg+5', 'PTS_Avg+10']
        pts_data = team_players[pts_cols].copy()
        pts_data = pts_data.sort_values('PTS_avg', ascending=False).head(15)
        pts_data.columns = ['Player', 'Avg PPG', 'Games', 'Avg-10', 'Avg-5', 'Avg+5', 'Avg+10']

        print("\n" + tabulate(pts_data, headers='keys', tablefmt='grid', showindex=False, floatfmt='.1f'))

        # =====================================================================
        # TABLE 2: ASSISTS HIT RATES
        # =====================================================================
        logger.info(f"\n>>> ASSISTS HIT RATES (% of games OVER line)")

        # Sort by APG descending, take top 15
        ast_cols = ['player_name', 'AST_avg', 'games', 'AST_Avg-10', 'AST_Avg-5', 'AST_Avg+5', 'AST_Avg+10']
        ast_data = team_players[ast_cols].copy()
        ast_data = ast_data.sort_values('AST_avg', ascending=False).head(15)
        ast_data.columns = ['Player', 'Avg APG', 'Games', 'Avg-10', 'Avg-5', 'Avg+5', 'Avg+10']

        print("\n" + tabulate(ast_data, headers='keys', tablefmt='grid', showindex=False, floatfmt='.1f'))

        # =====================================================================
        # TABLE 3: REBOUNDS HIT RATES
        # =====================================================================
        logger.info(f"\n>>> REBOUNDS HIT RATES (% of games OVER line)")

        # Sort by RPG descending, take top 15
        reb_cols = ['player_name', 'REB_avg', 'games', 'REB_Avg-10', 'REB_Avg-5', 'REB_Avg+5', 'REB_Avg+10']
        reb_data = team_players[reb_cols].copy()
        reb_data = reb_data.sort_values('REB_avg', ascending=False).head(15)
        reb_data.columns = ['Player', 'Avg RPG', 'Games', 'Avg-10', 'Avg-5', 'Avg+5', 'Avg+10']

        print("\n" + tabulate(reb_data, headers='keys', tablefmt='grid', showindex=False, floatfmt='.1f'))

    # =========================================================================
    # LEAGUE-WIDE SUMMARY STATISTICS
    # =========================================================================
    logger.info("\n" + "=" * 100)
    logger.info("LEAGUE-WIDE HIT RATE SUMMARY")
    logger.info("=" * 100)

    logger.info(f"\n>>> Average hit rates across all {len(all_players_df)} qualified players:")

    summary_data = []
    for stat_key, stat_info in stats_to_analyze.items():
        stat_summary = {'Stat': stat_info['name']}
        for offset_name in offsets.keys():
            col_name = f'{stat_key}_{offset_name}'
            avg_hit_rate = all_players_df[col_name].mean()
            stat_summary[offset_name] = avg_hit_rate
        summary_data.append(stat_summary)

    summary_df = pd.DataFrame(summary_data)
    print("\n" + tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.1f'))

    logger.info("\n>>> Interpretation Guide:")
    logger.info("    - Avg-10: Very conservative line (should have high hit rate ~80-90%)")
    logger.info("    - Avg-5: Conservative line (should have hit rate ~60-70%)")
    logger.info("    - Avg+5: Aggressive line (should have hit rate ~30-40%)")
    logger.info("    - Avg+10: Very aggressive line (should have low hit rate ~10-20%)")

    logger.info("\n>>> How to Use:")
    logger.info("    1. Look for players with 70%+ hit rate at Avg-5 (reliable for accumulators)")
    logger.info("    2. Compare actual hit rates vs expected to find value")
    logger.info("    3. Players with high Avg+5 rates (>40%) are outperforming their average")
    logger.info("    4. Low hit rates at conservative lines (<60% at Avg-5) suggest inconsistency")

    logger.info("\n✓ Line hit rate analysis complete!")

    # Return results for further analysis
    results = {
        'all_players': all_players_df,
        'by_team': results_by_team,
        'summary': summary_df
    }

    return results


def simulate_acca(df: pd.DataFrame, prop_analysis: pd.DataFrame, props: list = None, n_games: int = None):
    """
    Simplified accumulator simulator.

    Only requires:
    1. List of props (player name, stat, line, over/under)
    2. Number of games to simulate

    Args:
        df: Player game data
        prop_analysis: Player season stats
        props: List of dicts with keys: 'player', 'stat', 'line', 'over'
               Example: [
                   {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 20.5, 'over': True},
                   {'player': 'Mikal Bridges', 'stat': 'PTS', 'line': 18.5, 'over': True}
               ]
        n_games: Number of games to simulate (will randomly sample from dataset)

    Returns:
        results_df: DataFrame with simulation results
        props_df: DataFrame with individual prop results
    """
    logger.info("\n" + "=" * 100)
    logger.info("ACCUMULATOR SIMULATOR - SIMPLIFIED")
    logger.info("=" * 100)

    # Interactive input if not provided
    if props is None:
        print("\n" + "=" * 100)
        print("ACCUMULATOR SIMULATOR")
        print("=" * 100)
        print("\nAdd your player props one by one.")
        print("Available stats: PTS, AST, REB, BLK, STL, FG3M, TOV, MIN")
        print("=" * 100 + "\n")

        all_players = sorted(df['player_name'].unique())
        props = []

        prop_num = 1
        while True:
            print(f"\n{'='*80}")
            print(f"PROP {prop_num}")
            print(f"{'='*80}")

            # Get player name with validation
            while True:
                player_input = input("\nEnter player name (or 'search' to find players, 'done' to finish): ").strip()

                if player_input.lower() == 'done':
                    if len(props) == 0:
                        print("⚠️  You must add at least one prop!")
                        continue
                    break

                if player_input.lower() == 'search':
                    search_term = input("Enter search term: ").strip()
                    matches = [p for p in all_players if search_term.lower() in p.lower()]
                    if matches:
                        print(f"\n✓ Found {len(matches)} matching players:")
                        for i, player in enumerate(matches[:20], 1):
                            print(f"  {i}. {player}")
                        if len(matches) > 20:
                            print(f"  ... and {len(matches) - 20} more")
                    else:
                        print("❌ No players found.")
                    continue

                # Try exact match
                exact_matches = [p for p in all_players if p.lower() == player_input.lower()]
                if exact_matches:
                    player_name = exact_matches[0]
                    print(f"✓ Found: {player_name}")
                    break

                # Try partial match
                partial_matches = [p for p in all_players if player_input.lower() in p.lower()]
                if len(partial_matches) == 1:
                    player_name = partial_matches[0]
                    print(f"✓ Found: {player_name}")
                    break
                elif len(partial_matches) > 1:
                    print(f"\n⚠️  Multiple players found:")
                    for i, p in enumerate(partial_matches[:10], 1):
                        print(f"  {i}. {p}")
                    choice = input("\nEnter number to select: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= min(10, len(partial_matches)):
                        player_name = partial_matches[int(choice) - 1]
                        print(f"✓ Selected: {player_name}")
                        break
                    continue
                else:
                    print(f"❌ Player not found. Try 'search'.")
                    continue

            if player_input.lower() == 'done':
                break

            # Show player's season averages
            player_stats = prop_analysis[prop_analysis['player_name'] == player_name]
            if len(player_stats) > 0:
                stats_row = player_stats.iloc[0]
                print(f"\n📊 {player_name}'s Season Averages:")
                print(f"    PPG: {stats_row['PPG']:.1f}, APG: {stats_row['APG']:.1f}, RPG: {stats_row['RPG']:.1f}")
                print(f"    Games: {stats_row['games_played']}")

            # Get stat type
            while True:
                stat = input("\nEnter stat (PTS/AST/REB/BLK/STL/FG3M/TOV/MIN): ").strip().upper()
                if stat in ['PTS', 'AST', 'REB', 'BLK', 'STL', 'FG3M', 'TOV', 'MIN']:
                    break
                print("❌ Invalid stat.")

            # Get line
            while True:
                try:
                    line = float(input(f"Enter line for {stat}: ").strip())
                    break
                except ValueError:
                    print("❌ Invalid number.")

            # Get over/under
            while True:
                direction = input("Over or Under? (O/U): ").strip().upper()
                if direction in ['O', 'OVER']:
                    over = True
                    break
                elif direction in ['U', 'UNDER']:
                    over = False
                    break
                print("❌ Enter 'O' or 'U'")

            # Add prop
            props.append({
                'player': player_name,
                'stat': stat,
                'line': line,
                'over': over
            })

            direction_str = "Over" if over else "Under"
            print(f"\n✓ Added: {player_name} {direction_str} {line} {stat}")

            # Ask for another prop
            another = input("\nAdd another prop? (Y/N): ").strip().upper()
            if another != 'Y':
                break

            prop_num += 1

        if len(props) == 0:
            print("\n❌ No props added. Exiting.")
            return None, None

    # Get number of games if not provided
    if n_games is None:
        print("\n" + "=" * 100)
        print("SIMULATION SETTINGS")
        print("=" * 100)

        total_dates = df['game_date_only'].nunique()
        print(f"\nTotal game dates available in dataset: {total_dates}")

        while True:
            try:
                n_games = int(input(f"\nHow many games to simulate? (1-{total_dates}): ").strip())
                if 1 <= n_games <= total_dates:
                    break
                print(f"❌ Enter a number between 1 and {total_dates}")
            except ValueError:
                print("❌ Invalid number.")

    # Show summary
    print("\n" + "=" * 100)
    print("ACCUMULATOR SUMMARY")
    print("=" * 100)

    for i, prop in enumerate(props, 1):
        direction = "Over" if prop['over'] else "Under"
        print(f"{i}. {prop['player']} - {direction} {prop['line']} {prop['stat']}")

    print(f"\nTotal props: {len(props)}")
    print(f"Games to simulate: {n_games}")
    print("=" * 100)

    # Confirm
    confirm = input("\nRun simulation? (Y/N): ").strip().upper()
    if confirm != 'Y':
        print("\n❌ Simulation cancelled.")
        return None, None

    logger.info(f"\n>>> Running simulation with {len(props)} props across {n_games} games...")
    logger.info(f"    Props:")
    for prop in props:
        direction = "O" if prop['over'] else "U"
        logger.info(f"        {prop['player']} {prop['stat']} {direction}{prop['line']}")

    # Prepare data
    df['game_date_only'] = pd.to_datetime(df['game_date_only'])

    # Get all unique dates
    all_dates = sorted(df['game_date_only'].unique())

    # Randomly sample N dates for simulation
    import random
    random.seed(42)
    if n_games < len(all_dates):
        selected_dates = random.sample(list(all_dates), n_games)
    else:
        selected_dates = all_dates

    selected_dates = sorted(selected_dates)

    logger.info(f"\n>>> Randomly selected {len(selected_dates)} game dates for simulation")
    logger.info(f"    Date range: {selected_dates[0].date()} to {selected_dates[-1].date()}")

    # Run simulation
    results = []
    all_prop_results = []

    for game_num, test_date in enumerate(selected_dates, 1):
        games_that_night = df[df['game_date_only'] == test_date]

        logger.info(f"\n>>> Game {game_num}/{len(selected_dates)}: {test_date.date()}")
        logger.info(f"    {len(games_that_night)} games played that night")

        acca_hit = True
        props_hit = 0
        props_tested = 0
        prop_details = []

        for prop in props:
            # Find player's game(s) that night
            player_games = games_that_night[
                games_that_night['player_name'].str.contains(prop['player'], case=False, na=False)
            ]

            if len(player_games) == 0:
                # Player didn't play
                logger.info(f"        {prop['player']}: Did not play (SKIPPED)")
                continue

            # Use first game if multiple (shouldn't happen)
            player_game = player_games.iloc[0]
            result = player_game[prop['stat']]

            if prop['over']:
                hit = result > prop['line']
            else:
                hit = result < prop['line']

            props_tested += 1
            if hit:
                props_hit += 1
            else:
                acca_hit = False

            margin = result - prop['line']
            direction = "O" if prop['over'] else "U"
            status = "HIT ✓" if hit else "MISS ✗"

            logger.info(f"        {prop['player']} {prop['stat']} {direction}{prop['line']}: {result:.1f} - {status} (margin: {margin:+.1f})")

            # Get player consistency
            player_cv = prop_analysis[
                prop_analysis['player_name'].str.contains(prop['player'], case=False, na=False)
            ][f'{prop["stat"]}_CV'].values
            player_cv = player_cv[0] if len(player_cv) > 0 else np.nan

            prop_details.append({
                'game_num': game_num,
                'date': test_date,
                'player': prop['player'],
                'stat': prop['stat'],
                'line': prop['line'],
                'over': prop['over'],
                'result': result,
                'hit': hit,
                'margin': margin,
                'player_cv': player_cv
            })

        if props_tested == 0:
            logger.info(f"        NO PROPS TESTED (all players DNP) - VOID")
            continue

        hit_rate = (props_hit / props_tested * 100) if props_tested > 0 else 0
        status = "WIN ✓✓✓" if acca_hit else "LOSS ✗✗✗"

        logger.info(f"    RESULT: {status} ({props_hit}/{props_tested} props hit = {hit_rate:.1f}%)")

        results.append({
            'game_num': game_num,
            'date': test_date,
            'props_hit': props_hit,
            'props_tested': props_tested,
            'hit_rate': hit_rate,
            'acca_hit': acca_hit
        })

        all_prop_results.extend(prop_details)

    # Create DataFrames
    results_df = pd.DataFrame(results)
    props_df = pd.DataFrame(all_prop_results)

    # Display results
    logger.info("\n" + "=" * 100)
    logger.info("SIMULATION RESULTS SUMMARY")
    logger.info("=" * 100)

    if len(results_df) > 0:
        wins = results_df['acca_hit'].sum()
        total = len(results_df)
        win_rate = (wins / total * 100) if total > 0 else 0

        logger.info(f"\n>>> OVERALL PERFORMANCE:")
        logger.info(f"    Games simulated: {total}")
        logger.info(f"    Wins: {wins}")
        logger.info(f"    Losses: {total - wins}")
        logger.info(f"    Win rate: {win_rate:.1f}%")

        avg_hit_rate = results_df['hit_rate'].mean()
        logger.info(f"    Average prop hit rate: {avg_hit_rate:.1f}%")

        # Display results table
        from tabulate import tabulate
        display_results = results_df.copy()
        display_results['date'] = display_results['date'].dt.date
        display_results = display_results.round(2)

        logger.info("\n    GAME-BY-GAME RESULTS:")
        print(tabulate(display_results, headers='keys', tablefmt='grid', showindex=False))

        # Individual prop performance
        if len(props_df) > 0:
            logger.info("\n>>> PROP PERFORMANCE:")

            for prop in props:
                prop_results = props_df[props_df['player'].str.contains(prop['player'], case=False, na=False)]
                prop_results = prop_results[prop_results['stat'] == prop['stat']]

                if len(prop_results) > 0:
                    hit_rate = (prop_results['hit'].sum() / len(prop_results) * 100)
                    avg_margin = prop_results['margin'].mean()
                    direction = "O" if prop['over'] else "U"

                    logger.info(f"    {prop['player']} {prop['stat']} {direction}{prop['line']}:")
                    logger.info(f"        Hit rate: {hit_rate:.1f}% ({prop_results['hit'].sum()}/{len(prop_results)})")
                    logger.info(f"        Avg margin: {avg_margin:+.2f}")
                    logger.info(f"        CV: {prop_results['player_cv'].iloc[0]:.3f}")

    logger.info("\n✓ Simulation complete!")
    logger.info("=" * 100)

    return results_df, props_df


def interactive_acca_builder(df: pd.DataFrame, prop_analysis: pd.DataFrame):
    """
    Interactive accumulator builder with player name validation and bet testing.
    Allows user to input player names, stats, and lines, then backtests the acca.
    """
    logger.info("\n" + "=" * 100)
    logger.info("INTERACTIVE ACCUMULATOR BUILDER")
    logger.info("=" * 100)

    print("\n" + "=" * 100)
    print("WELCOME TO THE INTERACTIVE BET BUILDER")
    print("=" * 100)
    print("\nCreate your accumulator by adding legs with player props.")
    print("Players will be validated against the database.")
    print("\nAvailable stats: PTS (Points), AST (Assists), REB (Rebounds)")
    print("=" * 100 + "\n")

    # Get all available players
    all_players = sorted(df['player_name'].unique())

    acca_legs = []

    # Build legs
    leg_num = 1
    while True:
        print(f"\n{'='*80}")
        print(f"LEG {leg_num}")
        print(f"{'='*80}")

        leg_props = []

        # Add props to this leg
        prop_num = 1
        while True:
            print(f"\n--- Prop {prop_num} for Leg {leg_num} ---")

            # Get player name with validation
            while True:
                player_input = input("\nEnter player name (or 'search' to find players, 'done' to finish leg, 'cancel' to finish acca): ").strip()

                if player_input.lower() == 'cancel':
                    if len(acca_legs) == 0 and len(leg_props) == 0:
                        print("\n❌ No legs added. Exiting bet builder.")
                        return None, None
                    break

                if player_input.lower() == 'done':
                    if len(leg_props) == 0:
                        print("⚠️  You must add at least one prop to this leg!")
                        continue
                    break

                if player_input.lower() == 'search':
                    search_term = input("Enter search term (last name): ").strip()
                    matches = [p for p in all_players if search_term.lower() in p.lower()]
                    if matches:
                        print(f"\n✓ Found {len(matches)} matching players:")
                        for i, player in enumerate(matches[:20], 1):
                            print(f"  {i}. {player}")
                        if len(matches) > 20:
                            print(f"  ... and {len(matches) - 20} more")
                    else:
                        print("❌ No players found matching that search term.")
                    continue

                # Try to find exact match
                exact_matches = [p for p in all_players if p.lower() == player_input.lower()]
                if exact_matches:
                    player_name = exact_matches[0]
                    print(f"✓ Found exact match: {player_name}")
                    break

                # Try partial match
                partial_matches = [p for p in all_players if player_input.lower() in p.lower()]
                if len(partial_matches) == 1:
                    player_name = partial_matches[0]
                    print(f"✓ Found player: {player_name}")
                    break
                elif len(partial_matches) > 1:
                    print(f"\n⚠️  Multiple players found matching '{player_input}':")
                    for i, p in enumerate(partial_matches[:10], 1):
                        print(f"  {i}. {p}")
                    if len(partial_matches) > 10:
                        print(f"  ... and {len(partial_matches) - 10} more")

                    choice = input("\nEnter number to select, or press Enter to search again: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= min(10, len(partial_matches)):
                        player_name = partial_matches[int(choice) - 1]
                        print(f"✓ Selected: {player_name}")
                        break
                    continue
                else:
                    print(f"❌ Player '{player_input}' not found. Try 'search' to find players.")
                    continue

            if player_input.lower() == 'done':
                break
            if player_input.lower() == 'cancel':
                break

            # Get player's season averages for reference
            player_stats = prop_analysis[prop_analysis['player_name'] == player_name]
            if len(player_stats) > 0:
                stats_row = player_stats.iloc[0]
                print(f"\n📊 {player_name}'s Season Averages:")
                print(f"    PPG: {stats_row['PPG']:.1f} (CV: {stats_row.get('PTS_CV', 'N/A')})")
                print(f"    APG: {stats_row['APG']:.1f} (CV: {stats_row.get('AST_CV', 'N/A')})")
                print(f"    RPG: {stats_row['RPG']:.1f} (CV: {stats_row.get('REB_CV', 'N/A')})")
                print(f"    Games: {stats_row['games_played']}")

            # Get stat type
            while True:
                stat = input("\nEnter stat (PTS/AST/REB): ").strip().upper()
                if stat in ['PTS', 'AST', 'REB']:
                    break
                print("❌ Invalid stat. Use PTS, AST, or REB.")

            # Get line
            while True:
                try:
                    line = float(input(f"Enter line for {stat}: ").strip())
                    break
                except ValueError:
                    print("❌ Invalid number. Please enter a decimal number (e.g., 20.5)")

            # Get over/under
            while True:
                direction = input("Over or Under? (O/U): ").strip().upper()
                if direction in ['O', 'OVER']:
                    over = True
                    break
                elif direction in ['U', 'UNDER']:
                    over = False
                    break
                print("❌ Enter 'O' for Over or 'U' for Under")

            # Add prop
            leg_props.append({
                'player_last_name': player_name.split()[-1],
                'player_full_name': player_name,
                'stat': stat,
                'line': line,
                'over': over
            })

            direction_str = "Over" if over else "Under"
            print(f"\n✓ Added: {player_name} {direction_str} {line} {stat}")

            prop_num += 1

        if player_input.lower() == 'cancel':
            break

        # Add leg to acca
        if len(leg_props) > 0:
            leg_name = input(f"\nEnter name for Leg {leg_num} (or press Enter for default): ").strip()
            if not leg_name:
                leg_name = f"Leg {leg_num}"

            acca_legs.append({
                'name': leg_name,
                'props': leg_props
            })

            print(f"\n✓ Leg {leg_num} added with {len(leg_props)} prop(s)")

            # Ask if they want another leg
            another = input("\nAdd another leg? (Y/N): ").strip().upper()
            if another != 'Y':
                break

            leg_num += 1

    if len(acca_legs) == 0:
        print("\n❌ No legs added. Exiting bet builder.")
        return None, None

    # Show summary for confirmation
    print("\n" + "=" * 100)
    print("ACCUMULATOR SUMMARY")
    print("=" * 100)

    for leg_idx, leg in enumerate(acca_legs, 1):
        print(f"\n{leg['name']}:")
        for prop_idx, prop in enumerate(leg['props'], 1):
            direction = "Over" if prop['over'] else "Under"
            print(f"  {prop_idx}. {prop['player_full_name']} - {direction} {prop['line']} {prop['stat']}")

    print("\n" + "=" * 100)
    total_props = sum(len(leg['props']) for leg in acca_legs)
    print(f"Total: {len(acca_legs)} legs, {total_props} props")
    print("=" * 100)

    # Confirm
    confirm = input("\nConfirm and run backtest? (Y/N): ").strip().upper()
    if confirm != 'Y':
        print("\n❌ Backtest cancelled.")
        return None, None

    # Get date range
    print("\n" + "=" * 100)
    print("DATE RANGE SELECTION")
    print("=" * 100)

    min_date = df['game_date_only'].min()
    max_date = df['game_date_only'].max()
    print(f"\nAvailable data: {min_date} to {max_date}")

    use_all = input("\nUse all available dates? (Y/N): ").strip().upper()
    if use_all == 'Y':
        start_date = None
        end_date = None
    else:
        start_date = input("Enter start date (YYYY-MM-DD) or press Enter for earliest: ").strip()
        if not start_date:
            start_date = None

        end_date = input("Enter end date (YYYY-MM-DD) or press Enter for latest: ").strip()
        if not end_date:
            end_date = None

    # Create acca template
    acca_name = input("\nEnter name for this accumulator: ").strip()
    if not acca_name:
        acca_name = "Custom Acca"

    acca_template = {
        'name': acca_name,
        'legs': acca_legs
    }

    print(f"\n✓ Running backtest for '{acca_name}'...")

    # Run the backtest with the custom template
    return backtest_acca_across_dates_custom(df, prop_analysis, acca_template, start_date, end_date)


def backtest_acca_across_dates_custom(df: pd.DataFrame, prop_analysis: pd.DataFrame,
                                       acca_template: dict, start_date: str = None, end_date: str = None):
    """
    Backtest a custom accumulator template across a date range.
    Used by interactive builder.
    """
    logger.info("\n" + "=" * 100)
    logger.info("ACCUMULATOR BET BACKTESTING - CUSTOM ACCA")
    logger.info("=" * 100)

    # Convert dates
    df['game_date_only'] = pd.to_datetime(df['game_date_only'])

    if start_date is None:
        start_date = df['game_date_only'].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = df['game_date_only'].max()
    else:
        end_date = pd.to_datetime(end_date)

    logger.info(f"\n>>> Testing acca: {acca_template['name']}")
    logger.info(f">>> Date range: {start_date.date()} to {end_date.date()}")

    # Get all unique dates in range
    dates_in_range = df[(df['game_date_only'] >= start_date) &
                        (df['game_date_only'] <= end_date)]['game_date_only'].unique()
    dates_in_range = sorted(dates_in_range)

    logger.info(f">>> Total nights with games: {len(dates_in_range)}")

    # Results storage
    daily_results = []
    all_prop_results = []

    # Test each night
    for test_date in dates_in_range:
        games_that_night = df[df['game_date_only'] == test_date]

        leg_results_for_night = []

        for leg_idx, leg in enumerate(acca_template['legs']):
            props_evaluated = 0
            props_hit = 0
            props_in_leg = []

            for prop_template in leg['props']:
                # Find player by last name
                player_games = games_that_night[
                    games_that_night['player_name'].str.contains(
                        prop_template['player_last_name'],
                        case=False,
                        na=False
                    )
                ]

                if len(player_games) == 0:
                    # Player didn't play this night, skip
                    continue

                # Get the player's performance that night
                player_game = player_games.iloc[0]

                # Get the stat value
                if prop_template['stat'] == 'PTS':
                    result = player_game['PTS']
                elif prop_template['stat'] == 'AST':
                    result = player_game['AST']
                elif prop_template['stat'] == 'REB':
                    result = player_game['REB']
                else:
                    continue

                # Check if prop hit
                if prop_template['over']:
                    hit = result > prop_template['line']
                else:
                    hit = result < prop_template['line']

                margin = result - prop_template['line']

                props_evaluated += 1
                if hit:
                    props_hit += 1

                # Get player consistency metrics
                player_stats = prop_analysis[
                    prop_analysis['player_name'].str.contains(
                        prop_template['player_last_name'],
                        case=False,
                        na=False
                    )
                ]

                if len(player_stats) > 0:
                    player_row = player_stats.iloc[0]
                    if prop_template['stat'] == 'PTS':
                        cv = player_row.get('PTS_CV', np.nan)
                    elif prop_template['stat'] == 'AST':
                        cv = player_row.get('AST_CV', np.nan)
                    elif prop_template['stat'] == 'REB':
                        cv = player_row.get('REB_CV', np.nan)
                    else:
                        cv = np.nan
                else:
                    cv = np.nan

                props_in_leg.append({
                    'date': test_date,
                    'leg': leg_idx + 1,
                    'player': player_game['player_name'],
                    'stat': prop_template['stat'],
                    'line': prop_template['line'],
                    'result': result,
                    'margin': margin,
                    'hit': hit,
                    'cv': cv
                })

            # Evaluate leg (all props must hit)
            if props_evaluated > 0:
                leg_hit = (props_hit == props_evaluated)
                hit_rate = (props_hit / props_evaluated) * 100
            else:
                leg_hit = False
                hit_rate = 0

            leg_results_for_night.append({
                'date': test_date,
                'leg': leg_idx + 1,
                'leg_name': leg['name'],
                'props_evaluated': props_evaluated,
                'props_hit': props_hit,
                'hit_rate': hit_rate,
                'leg_won': leg_hit
            })

            all_prop_results.extend(props_in_leg)

        # Check if entire acca won (all legs must win)
        if len(leg_results_for_night) > 0:
            legs_evaluated = len(leg_results_for_night)
            legs_won = sum(1 for leg in leg_results_for_night if leg['leg_won'])
            acca_won = all(leg['leg_won'] for leg in leg_results_for_night)

            daily_results.append({
                'date': test_date,
                'legs_evaluated': legs_evaluated,
                'legs_won': legs_won,
                'acca_won': acca_won
            })

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
    accas_won = daily_df['acca_won'].sum()
    win_rate = (accas_won / total_nights) * 100

    logger.info(f"\n>>> OVERALL PERFORMANCE:")
    logger.info(f"    Total Nights Tested: {total_nights}")
    logger.info(f"    Accas Won: {accas_won}")
    logger.info(f"    Accas Lost: {total_nights - accas_won}")
    logger.info(f"    Win Rate: {win_rate:.1f}%")

    # Display results by date
    logger.info("\n>>> RESULTS BY DATE:")
    display_daily = daily_df.copy()
    display_daily['date'] = display_daily['date'].dt.strftime('%Y-%m-%d')
    display_daily.columns = ['Date', 'Legs Eval', 'Legs Won', 'Acca Won']

    print("\n" + tabulate(display_daily.head(30), headers='keys', tablefmt='grid', showindex=False))
    if len(display_daily) > 30:
        print(f"\n... and {len(display_daily) - 30} more nights")
    print()

    # Winning streaks analysis
    logger.info("\n>>> STREAK ANALYSIS:")
    daily_df['win_streak'] = (daily_df['acca_won'].astype(int)
                               .groupby((daily_df['acca_won'] != daily_df['acca_won'].shift()).cumsum())
                               .cumsum())
    daily_df['loss_streak'] = ((~daily_df['acca_won']).astype(int)
                                .groupby((daily_df['acca_won'] != daily_df['acca_won'].shift()).cumsum())
                                .cumsum())

    max_win_streak = daily_df['win_streak'].max()
    max_loss_streak = daily_df['loss_streak'].max()

    logger.info(f"    Longest Win Streak: {max_win_streak} night(s)")
    logger.info(f"    Longest Loss Streak: {max_loss_streak} night(s)")

    # Individual prop success rates
    logger.info("\n>>> INDIVIDUAL PROP SUCCESS RATES:")
    prop_summary = props_df.groupby(['player', 'stat']).agg({
        'hit': ['sum', 'count', 'mean'],
        'margin': 'mean',
        'cv': 'mean'
    }).round(2)
    prop_summary.columns = ['Hits', 'Total', 'Hit Rate', 'Avg Margin', 'Avg CV']
    prop_summary['Hit %'] = (prop_summary['Hit Rate'] * 100).round(1)
    prop_summary = prop_summary.sort_values('Hit %', ascending=False)

    print("\n" + tabulate(prop_summary.head(20), headers='keys', tablefmt='grid'))
    print()

    # Create visualizations (reuse from the date range function)
    logger.info("\n>>> Creating backtesting visualizations...")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Win rate over time
    ax1 = fig.add_subplot(gs[0, :])
    daily_df['date_str'] = daily_df['date'].dt.strftime('%m/%d')
    colors = ['green' if x else 'red' for x in daily_df['acca_won']]
    ax1.scatter(range(len(daily_df)), daily_df['acca_won'].astype(int), c=colors, s=100, alpha=0.6, edgecolor='black')
    ax1.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='50% baseline', alpha=0.5)
    ax1.set_xlabel('Night Index', fontweight='bold')
    ax1.set_ylabel('Won (1) / Lost (0)', fontweight='bold')
    ax1.set_title(f'{acca_template["name"]} - Win Rate: {win_rate:.1f}%', fontweight='bold', fontsize=14)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative wins
    ax2 = fig.add_subplot(gs[1, 0])
    daily_df['cumulative_wins'] = daily_df['acca_won'].cumsum()
    daily_df['cumulative_total'] = range(1, len(daily_df) + 1)
    ax2.plot(daily_df['cumulative_total'], daily_df['cumulative_wins'], linewidth=2, color='green', label='Wins')
    ax2.plot(daily_df['cumulative_total'], daily_df['cumulative_total'] * (win_rate/100),
             linewidth=2, linestyle='--', color='blue', label=f'Expected ({win_rate:.1f}%)')
    ax2.fill_between(daily_df['cumulative_total'], 0, daily_df['cumulative_wins'], alpha=0.3, color='green')
    ax2.set_xlabel('Nights Tested', fontweight='bold')
    ax2.set_ylabel('Cumulative Wins', fontweight='bold')
    ax2.set_title('Cumulative Wins Over Time', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Individual player success rates
    ax3 = fig.add_subplot(gs[1, 1:])
    player_success = props_df.groupby('player')['hit'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=True)
    player_success['pct'] = player_success['mean'] * 100
    y_pos = range(len(player_success))
    colors_player = ['green' if x >= 0.5 else 'red' for x in player_success['mean']]
    ax3.barh(y_pos, player_success['pct'], color=colors_player, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([p.split()[-1] for p in player_success.index], fontsize=9)
    ax3.axvline(50, color='blue', linestyle='--', linewidth=2, label='50% baseline')
    ax3.set_xlabel('Hit Rate (%)', fontweight='bold')
    ax3.set_title('Individual Player Prop Success Rates', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Margin distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(props_df['margin'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax4.axvline(props_df['margin'].mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {props_df["margin"].mean():.2f}')
    ax4.set_xlabel('Margin (Result - Line)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Margin Distribution', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Win rate by day of week
    ax5 = fig.add_subplot(gs[2, 1])
    daily_df['day_of_week'] = daily_df['date'].dt.day_name()
    day_win_rate = daily_df.groupby('day_of_week')['acca_won'].mean() * 100
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_win_rate = day_win_rate.reindex([d for d in day_order if d in day_win_rate.index])
    ax5.bar(range(len(day_win_rate)), day_win_rate.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(day_win_rate)))
    ax5.set_xticklabels([d[:3] for d in day_win_rate.index], rotation=45)
    ax5.axhline(win_rate, color='red', linestyle='--', linewidth=2, label=f'Overall: {win_rate:.1f}%')
    ax5.set_ylabel('Win Rate (%)', fontweight='bold')
    ax5.set_title('Win Rate by Day of Week', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Rolling win rate
    ax6 = fig.add_subplot(gs[2, 2])
    window = min(7, len(daily_df))
    daily_df['rolling_win_rate'] = daily_df['acca_won'].rolling(window=window, min_periods=1).mean() * 100
    ax6.plot(daily_df['rolling_win_rate'], linewidth=2, color='purple', label=f'{window}-night rolling avg')
    ax6.axhline(win_rate, color='red', linestyle='--', linewidth=2, label=f'Overall: {win_rate:.1f}%')
    ax6.fill_between(range(len(daily_df)), daily_df['rolling_win_rate'], win_rate,
                      where=daily_df['rolling_win_rate'] >= win_rate, alpha=0.3, color='green')
    ax6.fill_between(range(len(daily_df)), daily_df['rolling_win_rate'], win_rate,
                      where=daily_df['rolling_win_rate'] < win_rate, alpha=0.3, color='red')
    ax6.set_xlabel('Night Index', fontweight='bold')
    ax6.set_ylabel('Win Rate (%)', fontweight='bold')
    ax6.set_title(f'{window}-Night Rolling Win Rate', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Legs won distribution
    ax7 = fig.add_subplot(gs[3, 0])
    legs_won_counts = daily_df['legs_won'].value_counts().sort_index()
    ax7.bar(legs_won_counts.index, legs_won_counts.values, color='orange', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Number of Legs Won', fontweight='bold')
    ax7.set_ylabel('Frequency', fontweight='bold')
    ax7.set_title('Distribution of Legs Won per Night', fontweight='bold', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Prop hit rate by stat type
    ax8 = fig.add_subplot(gs[3, 1])
    stat_success = props_df.groupby('stat')['hit'].mean() * 100
    ax8.bar(stat_success.index, stat_success.values, color='teal', alpha=0.7, edgecolor='black')
    ax8.axhline(50, color='red', linestyle='--', linewidth=2, label='50% baseline')
    ax8.set_ylabel('Hit Rate (%)', fontweight='bold')
    ax8.set_title('Success Rate by Stat Type', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Consistency vs success
    ax9 = fig.add_subplot(gs[3, 2])
    valid_cv = props_df[props_df['cv'].notna()]
    colors_cv = ['green' if x else 'red' for x in valid_cv['hit']]
    ax9.scatter(valid_cv['cv'], valid_cv['margin'], c=colors_cv, alpha=0.6, s=80, edgecolor='black')
    ax9.axhline(0, color='blue', linestyle='--', linewidth=2, label='Break-even')
    ax9.axvline(0.40, color='orange', linestyle='--', linewidth=2, label='CV=0.40')
    ax9.set_xlabel('Player Consistency (CV)', fontweight='bold')
    ax9.set_ylabel('Margin', fontweight='bold')
    ax9.set_title('Player Consistency vs Margin', fontweight='bold', fontsize=12)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    fig.suptitle(f'{acca_template["name"]} Backtest - {start_date.date()} to {end_date.date()}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.show()

    logger.info("\n✓ Custom acca backtesting complete!")

    return daily_df, props_df


def backtest_acca_across_dates(df: pd.DataFrame, prop_analysis: pd.DataFrame,
                                start_date: str = None, end_date: str = None):
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

    # ============================================================================
    # 🎯 CUSTOMIZE YOUR ACCA HERE - EDIT THE TEMPLATE BELOW
    # ============================================================================
    #
    # This is where you define your custom accumulator bet to backtest!
    #
    # STRUCTURE:
    # - 'name': Name of your acca (for display purposes)
    # - 'legs': List of legs (each leg can have multiple props)
    #   - 'name': Name of the leg (for display)
    #   - 'props': List of player props in this leg
    #     - 'player_last_name': Last name of player (partial match supported)
    #     - 'stat': Stat type ('PTS', 'AST', 'REB', 'BLK', 'STL', 'FG3M', etc.)
    #     - 'line': The prop line value
    #     - 'over': True for over, False for under
    #
    # EXAMPLE: To test LaMelo Ball OVER 25.5 points:
    #   {'player_last_name': 'Ball', 'stat': 'PTS', 'line': 25.5, 'over': True}
    #
    # EXAMPLE: To test Towns UNDER 12.5 rebounds:
    #   {'player_last_name': 'Towns', 'stat': 'REB', 'line': 12.5, 'over': False}
    #
    # Add as many legs and props as you want!
    # ============================================================================

    acca_template = {
        'name': 'Sample 3-Leg Acca',  # ← EDIT: Change your acca name here
        'legs': [
            {
                'name': 'Leg 1 - Star Scorers',  # ← EDIT: Name of this leg
                'props': [
                    # ← EDIT: Add/remove/modify props below
                    {'player_last_name': 'Ball', 'stat': 'PTS', 'line': 20.5, 'over': True},
                    {'player_last_name': 'Bridges', 'stat': 'PTS', 'line': 18.5, 'over': True},
                ]
            },
            {
                'name': 'Leg 2 - All-Around Performance',
                'props': [
                    {'player_last_name': 'Brunson', 'stat': 'PTS', 'line': 22.5, 'over': True},
                    {'player_last_name': 'Towns', 'stat': 'REB', 'line': 10.5, 'over': True},
                    {'player_last_name': 'Wagner', 'stat': 'PTS', 'line': 19.5, 'over': True},
                ]
            },
            {
                'name': 'Leg 3 - Supporting Cast',
                'props': [
                    {'player_last_name': 'Porzingis', 'stat': 'PTS', 'line': 15.5, 'over': True},
                    {'player_last_name': 'Daniels', 'stat': 'AST', 'line': 4.5, 'over': True},
                    {'player_last_name': 'Murphy', 'stat': 'PTS', 'line': 13.5, 'over': True},
                ]
            }
            # ← ADD MORE LEGS HERE if needed
        ]
    }

    # Convert dates
    df['game_date_only'] = pd.to_datetime(df['game_date_only'])

    if start_date is None:
        start_date = df['game_date_only'].min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = df['game_date_only'].max()
    else:
        end_date = pd.to_datetime(end_date)

    logger.info(f"\n>>> Testing acca structure: {acca_template['name']}")
    logger.info(f">>> Date range: {start_date.date()} to {end_date.date()}")

    # Get all unique dates in range
    dates_in_range = df[(df['game_date_only'] >= start_date) &
                        (df['game_date_only'] <= end_date)]['game_date_only'].unique()
    dates_in_range = sorted(dates_in_range)

    logger.info(f">>> Total nights with games: {len(dates_in_range)}")

    # Results storage
    daily_results = []
    all_prop_results = []

    # Test each night
    for test_date in dates_in_range:
        games_that_night = df[df['game_date_only'] == test_date]

        leg_results_for_night = []

        for leg_idx, leg in enumerate(acca_template['legs']):
            props_evaluated = 0
            props_hit = 0
            props_in_leg = []

            for prop_template in leg['props']:
                # Find player by last name
                player_games = games_that_night[
                    games_that_night['player_name'].str.contains(
                        prop_template['player_last_name'],
                        case=False,
                        na=False
                    )
                ]

                if len(player_games) == 0:
                    # Player didn't play this night, skip
                    continue

                # Get the player's performance that night
                player_game = player_games.iloc[0]

                # Get the stat value
                if prop_template['stat'] == 'PTS':
                    result = player_game['PTS']
                elif prop_template['stat'] == 'AST':
                    result = player_game['AST']
                elif prop_template['stat'] == 'REB':
                    result = player_game['REB']
                else:
                    continue

                # Check if prop hit
                if prop_template['over']:
                    hit = result > prop_template['line']
                else:
                    hit = result < prop_template['line']

                margin = result - prop_template['line']

                props_evaluated += 1
                if hit:
                    props_hit += 1

                # Get player consistency metrics
                player_stats = prop_analysis[
                    prop_analysis['player_name'].str.contains(
                        prop_template['player_last_name'],
                        case=False,
                        na=False
                    )
                ]

                if len(player_stats) > 0:
                    player_row = player_stats.iloc[0]
                    if prop_template['stat'] == 'PTS':
                        cv = player_row.get('PTS_CV', np.nan)
                    elif prop_template['stat'] == 'AST':
                        cv = player_row.get('AST_CV', np.nan)
                    elif prop_template['stat'] == 'REB':
                        cv = player_row.get('REB_CV', np.nan)
                    else:
                        cv = np.nan
                else:
                    cv = np.nan

                props_in_leg.append({
                    'date': test_date,
                    'leg': leg_idx + 1,
                    'player': player_game['player_name'],
                    'stat': prop_template['stat'],
                    'line': prop_template['line'],
                    'result': result,
                    'margin': margin,
                    'hit': hit,
                    'cv': cv
                })

            # Evaluate leg (all props must hit)
            if props_evaluated > 0:
                leg_hit = (props_hit == props_evaluated)
                hit_rate = (props_hit / props_evaluated) * 100
            else:
                leg_hit = False
                hit_rate = 0

            leg_results_for_night.append({
                'date': test_date,
                'leg': leg_idx + 1,
                'leg_name': leg['name'],
                'props_evaluated': props_evaluated,
                'props_hit': props_hit,
                'hit_rate': hit_rate,
                'leg_won': leg_hit
            })

            all_prop_results.extend(props_in_leg)

        # Check if entire acca won (all legs must win)
        if len(leg_results_for_night) > 0:
            legs_evaluated = len(leg_results_for_night)
            legs_won = sum(1 for leg in leg_results_for_night if leg['leg_won'])
            acca_won = all(leg['leg_won'] for leg in leg_results_for_night)

            daily_results.append({
                'date': test_date,
                'legs_evaluated': legs_evaluated,
                'legs_won': legs_won,
                'acca_won': acca_won
            })

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
    accas_won = daily_df['acca_won'].sum()
    win_rate = (accas_won / total_nights) * 100

    logger.info(f"\n>>> OVERALL PERFORMANCE:")
    logger.info(f"    Total Nights Tested: {total_nights}")
    logger.info(f"    Accas Won: {accas_won}")
    logger.info(f"    Accas Lost: {total_nights - accas_won}")
    logger.info(f"    Win Rate: {win_rate:.1f}%")

    # Display results by date
    logger.info("\n>>> RESULTS BY DATE:")
    display_daily = daily_df.copy()
    display_daily['date'] = display_daily['date'].dt.strftime('%Y-%m-%d')
    display_daily.columns = ['Date', 'Legs Eval', 'Legs Won', 'Acca Won']

    print("\n" + tabulate(display_daily.head(30), headers='keys', tablefmt='grid', showindex=False))
    if len(display_daily) > 30:
        print(f"\n... and {len(display_daily) - 30} more nights")
    print()

    # Winning streaks analysis
    logger.info("\n>>> STREAK ANALYSIS:")
    daily_df['win_streak'] = (daily_df['acca_won'].astype(int)
                               .groupby((daily_df['acca_won'] != daily_df['acca_won'].shift()).cumsum())
                               .cumsum())
    daily_df['loss_streak'] = ((~daily_df['acca_won']).astype(int)
                                .groupby((daily_df['acca_won'] != daily_df['acca_won'].shift()).cumsum())
                                .cumsum())

    max_win_streak = daily_df['win_streak'].max()
    max_loss_streak = daily_df['loss_streak'].max()

    logger.info(f"    Longest Win Streak: {max_win_streak} night(s)")
    logger.info(f"    Longest Loss Streak: {max_loss_streak} night(s)")

    # Individual prop success rates
    logger.info("\n>>> INDIVIDUAL PROP SUCCESS RATES:")
    prop_summary = props_df.groupby(['player', 'stat']).agg({
        'hit': ['sum', 'count', 'mean'],
        'margin': 'mean',
        'cv': 'mean'
    }).round(2)
    prop_summary.columns = ['Hits', 'Total', 'Hit Rate', 'Avg Margin', 'Avg CV']
    prop_summary['Hit %'] = (prop_summary['Hit Rate'] * 100).round(1)
    prop_summary = prop_summary.sort_values('Hit %', ascending=False)

    print("\n" + tabulate(prop_summary.head(20), headers='keys', tablefmt='grid'))
    print()

    # VISUALIZATIONS
    logger.info("\n>>> Creating backtesting visualizations...")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Win rate over time
    ax1 = fig.add_subplot(gs[0, :])
    daily_df['date_str'] = daily_df['date'].dt.strftime('%m/%d')
    colors = ['green' if x else 'red' for x in daily_df['acca_won']]
    ax1.scatter(range(len(daily_df)), daily_df['acca_won'].astype(int), c=colors, s=100, alpha=0.6, edgecolor='black')
    ax1.axhline(0.5, color='blue', linestyle='--', linewidth=2, label='50% baseline', alpha=0.5)
    ax1.set_xlabel('Night Index', fontweight='bold')
    ax1.set_ylabel('Won (1) / Lost (0)', fontweight='bold')
    ax1.set_title(f'Accumulator Results Over Time - Win Rate: {win_rate:.1f}%', fontweight='bold', fontsize=14)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative wins
    ax2 = fig.add_subplot(gs[1, 0])
    daily_df['cumulative_wins'] = daily_df['acca_won'].cumsum()
    daily_df['cumulative_total'] = range(1, len(daily_df) + 1)
    ax2.plot(daily_df['cumulative_total'], daily_df['cumulative_wins'], linewidth=2, color='green', label='Wins')
    ax2.plot(daily_df['cumulative_total'], daily_df['cumulative_total'] * (win_rate/100),
             linewidth=2, linestyle='--', color='blue', label=f'Expected ({win_rate:.1f}%)')
    ax2.fill_between(daily_df['cumulative_total'], 0, daily_df['cumulative_wins'], alpha=0.3, color='green')
    ax2.set_xlabel('Nights Tested', fontweight='bold')
    ax2.set_ylabel('Cumulative Wins', fontweight='bold')
    ax2.set_title('Cumulative Wins Over Time', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Win rate by day of week
    ax3 = fig.add_subplot(gs[1, 1])
    daily_df['day_of_week'] = daily_df['date'].dt.day_name()
    day_win_rate = daily_df.groupby('day_of_week')['acca_won'].mean() * 100
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_win_rate = day_win_rate.reindex([d for d in day_order if d in day_win_rate.index])
    ax3.bar(range(len(day_win_rate)), day_win_rate.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(day_win_rate)))
    ax3.set_xticklabels([d[:3] for d in day_win_rate.index], rotation=45)
    ax3.axhline(win_rate, color='red', linestyle='--', linewidth=2, label=f'Overall: {win_rate:.1f}%')
    ax3.set_ylabel('Win Rate (%)', fontweight='bold')
    ax3.set_title('Win Rate by Day of Week', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Legs won distribution
    ax4 = fig.add_subplot(gs[1, 2])
    legs_won_counts = daily_df['legs_won'].value_counts().sort_index()
    ax4.bar(legs_won_counts.index, legs_won_counts.values, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Legs Won', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Distribution of Legs Won per Night', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Individual player success rates
    ax5 = fig.add_subplot(gs[2, :2])
    player_success = props_df.groupby('player')['hit'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=True)
    player_success['pct'] = player_success['mean'] * 100
    y_pos = range(len(player_success))
    colors_player = ['green' if x >= 0.5 else 'red' for x in player_success['mean']]
    ax5.barh(y_pos, player_success['pct'], color=colors_player, alpha=0.7, edgecolor='black')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([p.split()[-1] for p in player_success.index], fontsize=9)
    ax5.axvline(50, color='blue', linestyle='--', linewidth=2, label='50% baseline')
    ax5.set_xlabel('Hit Rate (%)', fontweight='bold')
    ax5.set_title('Individual Player Prop Success Rates', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Rolling win rate
    ax6 = fig.add_subplot(gs[2, 2])
    window = min(7, len(daily_df))
    daily_df['rolling_win_rate'] = daily_df['acca_won'].rolling(window=window, min_periods=1).mean() * 100
    ax6.plot(daily_df['rolling_win_rate'], linewidth=2, color='purple', label=f'{window}-night rolling avg')
    ax6.axhline(win_rate, color='red', linestyle='--', linewidth=2, label=f'Overall: {win_rate:.1f}%')
    ax6.fill_between(range(len(daily_df)), daily_df['rolling_win_rate'], win_rate,
                      where=daily_df['rolling_win_rate'] >= win_rate, alpha=0.3, color='green')
    ax6.fill_between(range(len(daily_df)), daily_df['rolling_win_rate'], win_rate,
                      where=daily_df['rolling_win_rate'] < win_rate, alpha=0.3, color='red')
    ax6.set_xlabel('Night Index', fontweight='bold')
    ax6.set_ylabel('Win Rate (%)', fontweight='bold')
    ax6.set_title(f'{window}-Night Rolling Win Rate', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Prop hit rate by stat type
    ax7 = fig.add_subplot(gs[3, 0])
    stat_success = props_df.groupby('stat')['hit'].mean() * 100
    ax7.bar(stat_success.index, stat_success.values, color='teal', alpha=0.7, edgecolor='black')
    ax7.axhline(50, color='red', linestyle='--', linewidth=2, label='50% baseline')
    ax7.set_ylabel('Hit Rate (%)', fontweight='bold')
    ax7.set_title('Success Rate by Stat Type', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Margin distribution
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.hist(props_df['margin'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax8.axvline(props_df['margin'].mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {props_df["margin"].mean():.2f}')
    ax8.set_xlabel('Margin (Result - Line)', fontweight='bold')
    ax8.set_ylabel('Frequency', fontweight='bold')
    ax8.set_title('Margin Distribution Across All Props', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Consistency vs success
    ax9 = fig.add_subplot(gs[3, 2])
    valid_cv = props_df[props_df['cv'].notna()]
    colors_cv = ['green' if x else 'red' for x in valid_cv['hit']]
    ax9.scatter(valid_cv['cv'], valid_cv['margin'], c=colors_cv, alpha=0.6, s=80, edgecolor='black')
    ax9.axhline(0, color='blue', linestyle='--', linewidth=2, label='Break-even')
    ax9.axvline(0.40, color='orange', linestyle='--', linewidth=2, label='CV=0.40')
    ax9.set_xlabel('Player Consistency (CV)', fontweight='bold')
    ax9.set_ylabel('Margin', fontweight='bold')
    ax9.set_title('Player Consistency vs Margin', fontweight='bold', fontsize=12)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    fig.suptitle(f'Accumulator Backtesting Analysis - {start_date.date()} to {end_date.date()}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.show()

    logger.info("\n✓ Date range backtesting complete!")

    return daily_df, props_df


def analyze_historical_accas(df: pd.DataFrame, prop_analysis: pd.DataFrame):
    """
    Analyze historical accumulator bets with detailed metrics and visualizations.

    SIMPLIFIED INPUT FORMAT - Only requires:
    - 'date': Game date (YYYY-MM-DD)
    - 'props': List of props with player, stat, line, over
    - 'odds': (OPTIONAL) Odds for the leg, defaults to 1.0

    Results are automatically looked up from the dataframe!
    """
    logger.info("\n" + "=" * 100)
    logger.info("ACCUMULATOR BET BACKTESTING - HISTORICAL ANALYSIS")
    logger.info("=" * 100)
    logger.info("\n>>> Analyzing historical acca performance with player consistency data...")

    # ============================================================================
    # 🎯 SIMPLIFIED ACCUMULATOR INPUT - EDIT BELOW
    # ============================================================================
    #
    # ONLY REQUIRED FIELDS:
    # - 'date': Date of the game (YYYY-MM-DD format)
    # - 'props': List of player props
    #   - 'player': Player name (full or partial, last name works)
    #   - 'stat': Stat type ('PTS', 'AST', 'REB', 'BLK', 'STL', 'FG3M', 'TOV', 'MIN')
    #   - 'line': The prop line value
    #   - 'over': True for over, False for under
    #
    # OPTIONAL FIELDS:
    # - 'odds': Odds for this leg (defaults to 1.0 if not provided)
    #
    # NO NEED FOR: 'result', 'match', 'time', 'leg' - these are automatic!
    # ============================================================================

    historical_accas = [
        {
            'date': '2024-11-22',
            'odds': 2.70,  # Optional
            'props': [
                {'player': 'Kon Knueppel', 'stat': 'PTS', 'line': 17.5, 'over': True},
                {'player': 'Miles Bridges', 'stat': 'PTS', 'line': 11.5, 'over': True},
                {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 12.5, 'over': True},
            ]
        },
        {
            'date': '2024-11-22',
            'odds': 4.90,  # Optional
            'props': [
                {'player': 'Wendell Carter', 'stat': 'PTS', 'line': 7.5, 'over': True},
                {'player': 'Mikal Bridges', 'stat': 'PTS', 'line': 11.5, 'over': True},
                {'player': 'Franz Wagner', 'stat': 'PTS', 'line': 17.5, 'over': True},
                {'player': 'Jalen Brunson', 'stat': 'PTS', 'line': 21.5, 'over': True},
                {'player': 'Karl-Anthony Towns', 'stat': 'PTS', 'line': 13.5, 'over': True},
                {'player': 'Jalen Suggs', 'stat': 'PTS', 'line': 10.5, 'over': True},
                {'player': 'Anthony Black', 'stat': 'PTS', 'line': 7.5, 'over': True},
            ]
        },
        {
            'date': '2024-11-22',
            'odds': 3.80,  # Optional
            'props': [
                {'player': 'Trey Murphy', 'stat': 'PTS', 'line': 12.5, 'over': True},
                {'player': 'Kristaps Porzingis', 'stat': 'PTS', 'line': 9.5, 'over': True},
                {'player': 'Dyson Daniels', 'stat': 'REB', 'line': 3.5, 'over': True},
                {'player': 'Dyson Daniels', 'stat': 'AST', 'line': 3.5, 'over': True},
                {'player': 'Dyson Daniels', 'stat': 'PTS', 'line': 7.5, 'over': True},
                {'player': 'Nickeil Alexander-Walker', 'stat': 'PTS', 'line': 14.5, 'over': True},
                {'player': 'Jose Alvarado', 'stat': 'PTS', 'line': 2.5, 'over': True},
            ]
        }
    ]

    # Convert date column to datetime for matching
    df['game_date_only'] = pd.to_datetime(df['game_date_only'])

    # Analyze each leg
    leg_results = []
    prop_results = []

    for leg_num, acca in enumerate(historical_accas, 1):
        # Get odds (default to 1.0 if not provided)
        odds = acca.get('odds', 1.0)

        # Convert date string to datetime
        acca_date = pd.to_datetime(acca['date'])

        # Get games on this date
        games_that_night = df[df['game_date_only'] == acca_date]

        if len(games_that_night) == 0:
            logger.warning(f"⚠️  No games found on {acca['date']}, skipping leg {leg_num}")
            continue

        logger.info(f"\n>>> Analyzing Leg {leg_num}: {acca['date']}")
        logger.info(f"    Games that night: {len(games_that_night['gameId'].nunique())}")

        leg_hit = True
        props_hit = 0
        total_props = len(acca['props'])
        props_evaluated = 0

        for prop in acca['props']:
            # Find player by name (supports partial matching)
            player_last_name = prop['player'].split()[-1]
            player_games = games_that_night[
                games_that_night['player_name'].str.contains(player_last_name, case=False, na=False)
            ]

            if len(player_games) == 0:
                logger.warning(f"    ⚠️  {prop['player']} did not play on {acca['date']}, skipping prop")
                continue

            # Get the player's performance that night
            player_game = player_games.iloc[0]
            player_full_name = player_game['player_name']

            # Look up the stat value based on stat type
            stat_map = {
                'PTS': 'PTS',
                'AST': 'AST',
                'REB': 'REB',
                'BLK': 'BLK',
                'STL': 'STL',
                'FG3M': 'FG3M',
                'TOV': 'TOV',
                'MIN': 'MIN'
            }

            if prop['stat'] not in stat_map:
                logger.warning(f"    ⚠️  Unsupported stat type: {prop['stat']}, skipping")
                continue

            result = player_game[stat_map[prop['stat']]]

            # Determine if prop hit
            if prop['over']:
                hit = result > prop['line']
            else:
                hit = result < prop['line']

            margin = result - prop['line']

            props_evaluated += 1

            # Get historical player stats for consistency metrics
            player_stats = prop_analysis[
                prop_analysis['player_name'].str.contains(player_last_name, case=False, na=False)
            ]

            if len(player_stats) > 0:
                player_row = player_stats.iloc[0]
                if prop['stat'] == 'PTS':
                    avg = player_row['PPG']
                    cv = player_row.get('PTS_CV', np.nan)
                elif prop['stat'] == 'AST':
                    avg = player_row['APG']
                    cv = player_row.get('AST_CV', np.nan)
                elif prop['stat'] == 'REB':
                    avg = player_row['RPG']
                    cv = player_row.get('REB_CV', np.nan)
                else:
                    avg = np.nan
                    cv = np.nan
            else:
                avg = np.nan
                cv = np.nan

            prop_results.append({
                'leg': leg_num,
                'player': player_full_name,
                'stat': prop['stat'],
                'line': prop['line'],
                'result': result,
                'margin': margin,
                'hit': hit,
                'player_avg': avg,
                'player_cv': cv
            })

            # Log the prop result
            hit_symbol = "✓" if hit else "✗"
            logger.info(f"    {player_full_name} {prop['stat']} {'O' if prop['over'] else 'U'}{prop['line']}: "
                       f"{result:.1f} - {hit_symbol} (margin: {margin:+.1f})")

            if hit:
                props_hit += 1
            else:
                leg_hit = False

        # Skip leg if no props were evaluated
        if props_evaluated == 0:
            logger.warning(f"    ⚠️  No props could be evaluated for leg {leg_num}, skipping")
            continue

        hit_rate = (props_hit / props_evaluated) * 100
        implied_prob = 1 / odds if odds > 0 else 1.0
        expected_hit_rate = implied_prob * 100

        leg_results.append({
            'leg': leg_num,
            'date': acca['date'],
            'odds': odds,
            'total_props': props_evaluated,
            'props_hit': props_hit,
            'hit_rate': hit_rate,
            'leg_won': leg_hit,
            'implied_prob': implied_prob,
            'expected_hit_rate': expected_hit_rate
        })

        # Log leg result
        if leg_hit:
            logger.info(f"    ✓✓✓ LEG WON ({props_hit}/{props_evaluated} props hit = {hit_rate:.1f}%)")
        else:
            logger.info(f"    ✗✗✗ LEG LOST ({props_hit}/{props_evaluated} props hit = {hit_rate:.1f}%)")

    # Convert to DataFrames
    leg_df = pd.DataFrame(leg_results)
    prop_df = pd.DataFrame(prop_results)

    # Display leg summary
    logger.info("\n" + "=" * 100)
    logger.info("ACCUMULATOR LEG SUMMARY")
    logger.info("=" * 100)

    display_table = leg_df[[
        'leg', 'total_props', 'props_hit', 'hit_rate', 'leg_won', 'odds', 'implied_prob'
    ]].copy()

    display_table.columns = ['Leg', 'Props', 'Hit', 'Hit%', 'Won', 'Odds', 'Implied Prob']
    display_table['Hit%'] = display_table['Hit%'].round(1)
    display_table['Implied Prob'] = (display_table['Implied Prob'] * 100).round(1)

    print("\n" + tabulate(display_table, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    print()

    # Overall acca results
    total_legs = len(leg_df)
    legs_won = leg_df['leg_won'].sum()
    acca_win_rate = (legs_won / total_legs) * 100
    total_odds = leg_df['odds'].prod()
    total_implied_prob = leg_df['implied_prob'].prod()

    logger.info("\n>>> OVERALL ACCUMULATOR PERFORMANCE:")
    logger.info(f"    Total Legs: {total_legs}")
    logger.info(f"    Legs Won: {legs_won}")
    logger.info(f"    Legs Lost: {total_legs - legs_won}")
    logger.info(f"    Acca Win Rate: {acca_win_rate:.1f}%")
    logger.info(f"    Combined Odds: {total_odds:.2f}")
    logger.info(f"    Implied Probability: {total_implied_prob*100:.2f}%")
    if legs_won == total_legs:
        logger.info(f"    ✓ ACCA WON! Payout: {total_odds:.2f}x")
    else:
        logger.info(f"    ✗ ACCA LOST")

    # Individual prop analysis
    logger.info("\n" + "=" * 100)
    logger.info("INDIVIDUAL PROP PERFORMANCE")
    logger.info("=" * 100)

    prop_display = prop_df[[
        'leg', 'player', 'stat', 'line', 'result', 'margin', 'hit', 'player_avg', 'player_cv'
    ]].copy()

    prop_display.columns = ['Leg', 'Player', 'Stat', 'Line', 'Result', 'Margin', 'Hit', 'Avg', 'CV']
    prop_display = prop_display.round(2)

    print("\n" + tabulate(prop_display, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
    print()

    # Hit rate by stat type
    logger.info("\n>>> HIT RATE BY STAT TYPE:")
    stat_hit_rate = prop_df.groupby('stat').agg({
        'hit': ['sum', 'count', 'mean']
    }).round(3)
    stat_hit_rate.columns = ['Hits', 'Total', 'Hit Rate']
    stat_hit_rate['Hit %'] = (stat_hit_rate['Hit Rate'] * 100).round(1)

    print("\n" + tabulate(stat_hit_rate, headers='keys', tablefmt='grid', floatfmt='.2f'))
    print()

    # Margin analysis
    logger.info("\n>>> MARGIN ANALYSIS (Result - Line):")
    margin_stats = prop_df['margin'].describe()
    logger.info(f"    Mean Margin: {margin_stats['mean']:.2f}")
    logger.info(f"    Median Margin: {margin_stats['50%']:.2f}")
    logger.info(f"    Min Margin: {margin_stats['min']:.2f}")
    logger.info(f"    Max Margin: {margin_stats['max']:.2f}")

    # VISUALIZATIONS
    logger.info("\n>>> Creating accumulator analysis visualizations...")

    # Figure 1: Overview dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Hit rate by leg
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(leg_df['leg'], leg_df['hit_rate'], color=['green' if x else 'red' for x in leg_df['leg_won']],
            alpha=0.7, edgecolor='black')
    ax1.axhline(100, color='blue', linestyle='--', label='100% (All Hit)', alpha=0.5)
    ax1.set_xlabel('Leg', fontweight='bold')
    ax1.set_ylabel('Hit Rate (%)', fontweight='bold')
    ax1.set_title('Hit Rate by Accumulator Leg', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Props hit vs total by leg
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(leg_df))
    width = 0.35
    ax2.bar(x - width/2, leg_df['total_props'], width, label='Total Props', alpha=0.7, color='lightblue', edgecolor='black')
    ax2.bar(x + width/2, leg_df['props_hit'], width, label='Props Hit', alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Leg', fontweight='bold')
    ax2.set_ylabel('Number of Props', fontweight='bold')
    ax2.set_title('Props Hit vs Total Props by Leg', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(leg_df['leg'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Margin distribution
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if x else 'red' for x in prop_df['hit']]
    ax3.scatter(range(len(prop_df)), prop_df['margin'], c=colors, alpha=0.6, s=100, edgecolor='black')
    ax3.axhline(0, color='blue', linestyle='--', linewidth=2, label='Break-even line')
    ax3.set_xlabel('Prop Index', fontweight='bold')
    ax3.set_ylabel('Margin (Result - Line)', fontweight='bold')
    ax3.set_title('Margin Above/Below Line for Each Prop', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Hit rate by stat type
    ax4 = fig.add_subplot(gs[1, 0])
    stat_summary = prop_df.groupby('stat')['hit'].mean() * 100
    ax4.bar(stat_summary.index, stat_summary.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axhline(50, color='red', linestyle='--', label='50% baseline', alpha=0.5)
    ax4.set_xlabel('Stat Type', fontweight='bold')
    ax4.set_ylabel('Hit Rate (%)', fontweight='bold')
    ax4.set_title('Hit Rate by Stat Type', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Player consistency vs hit
    ax5 = fig.add_subplot(gs[1, 1])
    valid_cv = prop_df[prop_df['player_cv'].notna()]
    colors = ['green' if x else 'red' for x in valid_cv['hit']]
    ax5.scatter(valid_cv['player_cv'], valid_cv['margin'], c=colors, alpha=0.6, s=100, edgecolor='black')
    ax5.axhline(0, color='blue', linestyle='--', linewidth=2)
    ax5.axvline(0.40, color='orange', linestyle='--', linewidth=2, label='CV=0.40 threshold')
    ax5.set_xlabel('Player Consistency (CV - Lower = Better)', fontweight='bold')
    ax5.set_ylabel('Margin Above/Below Line', fontweight='bold')
    ax5.set_title('Player Consistency vs Prop Success', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Result vs line scatter
    ax6 = fig.add_subplot(gs[1, 2])
    colors = ['green' if x else 'red' for x in prop_df['hit']]
    ax6.scatter(prop_df['line'], prop_df['result'], c=colors, alpha=0.6, s=100, edgecolor='black')
    max_val = max(prop_df['line'].max(), prop_df['result'].max())
    ax6.plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='Break-even line')
    ax6.set_xlabel('Line (Prop Target)', fontweight='bold')
    ax6.set_ylabel('Result (Actual Performance)', fontweight='bold')
    ax6.set_title('Actual Result vs Prop Line', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Margin histogram
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(prop_df['margin'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax7.axvline(prop_df['margin'].mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {prop_df["margin"].mean():.2f}')
    ax7.set_xlabel('Margin (Result - Line)', fontweight='bold')
    ax7.set_ylabel('Frequency', fontweight='bold')
    ax7.set_title('Distribution of Margins', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Cumulative props by leg
    ax8 = fig.add_subplot(gs[2, 1])
    cumulative_hit = leg_df['props_hit'].cumsum()
    cumulative_total = leg_df['total_props'].cumsum()
    ax8.plot(leg_df['leg'], cumulative_total, marker='o', linewidth=2, label='Total Props', color='blue')
    ax8.plot(leg_df['leg'], cumulative_hit, marker='o', linewidth=2, label='Props Hit', color='green')
    ax8.fill_between(leg_df['leg'], cumulative_hit, cumulative_total, alpha=0.3, color='red', label='Missed')
    ax8.set_xlabel('Leg', fontweight='bold')
    ax8.set_ylabel('Cumulative Props', fontweight='bold')
    ax8.set_title('Cumulative Props Across Legs', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Odds vs implied probability
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.bar(leg_df['leg'], leg_df['implied_prob'] * 100, alpha=0.7, color='orange', edgecolor='black',
            label='Implied Probability')
    ax9.bar(leg_df['leg'], leg_df['hit_rate'], alpha=0.5, color='green', edgecolor='black',
            label='Actual Hit Rate')
    ax9.set_xlabel('Leg', fontweight='bold')
    ax9.set_ylabel('Percentage (%)', fontweight='bold')
    ax9.set_title('Implied Probability vs Actual Hit Rate', fontweight='bold', fontsize=12)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Accumulator Bet Analysis Dashboard - Historical Performance',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.show()

    # Figure 2: Player-specific analysis
    logger.info("\n>>> Creating player-specific performance analysis...")

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Player performance by leg
    ax1 = axes[0, 0]
    for leg in prop_df['leg'].unique():
        leg_data = prop_df[prop_df['leg'] == leg]
        hit_rate = leg_data['hit'].mean() * 100
        ax1.bar(leg, hit_rate, alpha=0.7, edgecolor='black')

    ax1.set_xlabel('Leg', fontweight='bold')
    ax1.set_ylabel('Hit Rate (%)', fontweight='bold')
    ax1.set_title('Average Hit Rate per Leg', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Margin by player
    ax2 = axes[0, 1]
    player_margins = prop_df.groupby('player')['margin'].mean().sort_values()
    colors_player = ['green' if x > 0 else 'red' for x in player_margins.values]
    ax2.barh(range(len(player_margins)), player_margins.values, color=colors_player, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(player_margins)))
    ax2.set_yticklabels([p.split()[-1] for p in player_margins.index], fontsize=8)  # Last name only
    ax2.axvline(0, color='blue', linestyle='--', linewidth=2)
    ax2.set_xlabel('Average Margin', fontweight='bold')
    ax2.set_title('Average Margin by Player (Result - Line)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')

    # Player CV distribution
    ax3 = axes[1, 0]
    valid_players = prop_df[prop_df['player_cv'].notna()]
    ax3.hist(valid_players['player_cv'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(0.40, color='red', linestyle='--', linewidth=2, label='CV=0.40 (Good threshold)')
    ax3.set_xlabel('Player Coefficient of Variation', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Distribution of Player Consistency (CV)', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Success rate by number of props in leg
    ax4 = axes[1, 1]
    leg_size_analysis = leg_df.groupby('total_props').agg({
        'hit_rate': 'mean',
        'leg_won': 'mean'
    })
    ax4.plot(leg_size_analysis.index, leg_size_analysis['hit_rate'], marker='o', linewidth=2,
             markersize=10, label='Avg Hit Rate (%)', color='blue')
    ax4.plot(leg_size_analysis.index, leg_size_analysis['leg_won'] * 100, marker='s', linewidth=2,
             markersize=10, label='Leg Win Rate (%)', color='green')
    ax4.set_xlabel('Number of Props in Leg', fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontweight='bold')
    ax4.set_title('Performance by Leg Size', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig2.suptitle('Player-Level Accumulator Analysis',
                  fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # KEY INSIGHTS
    logger.info("\n" + "=" * 100)
    logger.info("KEY INSIGHTS & RECOMMENDATIONS")
    logger.info("=" * 100)

    # Most reliable players
    reliable_players = prop_df[prop_df['hit'] == True].groupby('player').size().sort_values(ascending=False)
    if len(reliable_players) > 0:
        logger.info("\n>>> MOST RELIABLE PLAYERS (All Props Hit):")
        for player, count in reliable_players.head(5).items():
            logger.info(f"    {player}: {count} prop(s) hit")

    # Biggest margins
    logger.info("\n>>> BIGGEST WINS (Highest Positive Margin):")
    top_margins = prop_df.nlargest(5, 'margin')[['player', 'stat', 'line', 'result', 'margin']]
    for _, row in top_margins.iterrows():
        logger.info(f"    {row['player']} - {row['stat']}: {row['result']:.0f} (Line: {row['line']}, +{row['margin']:.1f})")

    # Close calls
    logger.info("\n>>> CLOSE CALLS (Margin < 2):")
    close_calls = prop_df[prop_df['margin'].abs() < 2][['player', 'stat', 'line', 'result', 'margin', 'hit']]
    for _, row in close_calls.iterrows():
        status = "HIT ✓" if row['hit'] else "MISS ✗"
        logger.info(f"    {row['player']} - {row['stat']}: {row['result']:.0f} vs {row['line']} ({status}, margin: {row['margin']:.1f})")

    # Consistency analysis
    logger.info("\n>>> CONSISTENCY ANALYSIS:")
    consistent_hits = prop_df[(prop_df['player_cv'] < 0.40) & (prop_df['hit'] == True)]
    inconsistent_hits = prop_df[(prop_df['player_cv'] >= 0.40) & (prop_df['hit'] == True)]
    logger.info(f"    Props hit from consistent players (CV<0.40): {len(consistent_hits)}")
    logger.info(f"    Props hit from inconsistent players (CV≥0.40): {len(inconsistent_hits)}")

    logger.info("\n✓ Accumulator backtesting complete!")

    return leg_df, prop_df


def test_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test each numeric variable against multiple distributions to find best fit.
    Uses Kolmogorov-Smirnov, Anderson-Darling, and Chi-Square tests.

    Returns DataFrame with distribution test results for each variable.
    """
    from scipy import stats
    from scipy.stats import kstest, anderson, chisquare, normaltest, jarque_bera
    import warnings
    warnings.filterwarnings('ignore')

    logger.info("\n" + "=" * 100)
    logger.info("DISTRIBUTION TESTING - STATISTICAL GOODNESS-OF-FIT ANALYSIS")
    logger.info("=" * 100)

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Distributions to test
    distributions_to_test = {
        'Normal': stats.norm,
        'Lognormal': stats.lognorm,
        'Exponential': stats.expon,
        'Gamma': stats.gamma,
        'Beta': stats.beta,
        'Weibull': stats.weibull_min,
        'Poisson': stats.poisson,
        'Negative Binomial': stats.nbinom
    }

    logger.info(f"\n>>> Testing {len(numeric_cols)} variables against {len(distributions_to_test)} distributions...")
    logger.info(f"    Using: Kolmogorov-Smirnov, Anderson-Darling, Shapiro-Wilk, Jarque-Bera tests")

    results = []

    for col in numeric_cols:
        data = df[col].dropna()

        if len(data) < 3:
            continue

        logger.info(f"\n>>> Testing variable: {col}")
        logger.info(f"    Sample size: n={len(data):,}")
        logger.info(f"    Range: [{data.min():.3f}, {data.max():.3f}]")
        logger.info(f"    Mean: {data.mean():.3f}, Std: {data.std():.3f}")
        logger.info(f"    Skewness: {stats.skew(data):.3f}, Kurtosis: {stats.kurtosis(data):.3f}")

        best_dist = None
        best_ks_stat = np.inf
        best_ks_pval = 0

        dist_results = {}

        for dist_name, dist_func in distributions_to_test.items():
            try:
                # Skip distributions not suitable for the data
                if dist_name == 'Lognormal' and (data <= 0).any():
                    continue
                if dist_name == 'Beta' and ((data < 0).any() or (data > 1).any()):
                    continue
                if dist_name in ['Poisson', 'Negative Binomial'] and not np.allclose(data, data.astype(int)):
                    continue

                # Fit distribution
                if dist_name == 'Normal':
                    params = dist_func.fit(data)
                    fitted = dist_func(*params)
                elif dist_name == 'Exponential':
                    params = dist_func.fit(data, floc=0)
                    fitted = dist_func(*params)
                elif dist_name == 'Poisson':
                    params = (data.mean(),)
                    fitted = dist_func(*params)
                else:
                    params = dist_func.fit(data)
                    fitted = dist_func(*params)

                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = kstest(data, lambda x: fitted.cdf(x))

                # Anderson-Darling test (only for normal, exponential, logistic)
                ad_stat = None
                ad_critical = None
                if dist_name in ['Normal', 'Exponential']:
                    ad_result = anderson(data, dist=dist_name.lower())
                    ad_stat = ad_result.statistic
                    ad_critical = ad_result.critical_values[2]  # 5% significance level

                dist_results[dist_name] = {
                    'ks_stat': ks_stat,
                    'ks_pval': ks_pval,
                    'ad_stat': ad_stat,
                    'ad_critical': ad_critical,
                    'params': params
                }

                # Track best fit (lowest KS statistic, highest p-value)
                if ks_pval > best_ks_pval:
                    best_ks_pval = ks_pval
                    best_ks_stat = ks_stat
                    best_dist = dist_name

            except Exception as e:
                logger.debug(f"        {dist_name}: Failed to fit ({str(e)[:50]})")
                continue

        # Normality tests
        shapiro_stat, shapiro_pval = stats.shapiro(data[:5000]) if len(data) > 5000 else stats.shapiro(data)
        jb_stat, jb_pval = jarque_bera(data)

        # Determine best distribution
        if best_dist:
            logger.info(f"    BEST FIT: {best_dist}")
            logger.info(f"        KS statistic: {best_ks_stat:.4f}")
            logger.info(f"        KS p-value: {best_ks_pval:.4f}")
            if dist_results[best_dist]['ad_stat']:
                logger.info(f"        AD statistic: {dist_results[best_dist]['ad_stat']:.4f}")

        logger.info(f"    NORMALITY TESTS:")
        logger.info(f"        Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_pval:.4f}")
        logger.info(f"        Jarque-Bera: JB={jb_stat:.4f}, p={jb_pval:.4f}")

        # Interpretation
        is_normal = (shapiro_pval > 0.05) and (jb_pval > 0.05)
        interpretation = "NORMAL" if is_normal else "NON-NORMAL"
        logger.info(f"    INTERPRETATION: {interpretation} (α=0.05)")

        results.append({
            'Variable': col,
            'N': len(data),
            'Mean': data.mean(),
            'Std': data.std(),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data),
            'Best_Distribution': best_dist if best_dist else 'Unknown',
            'KS_Statistic': best_ks_stat if best_ks_stat != np.inf else np.nan,
            'KS_P_Value': best_ks_pval,
            'Shapiro_W': shapiro_stat,
            'Shapiro_P': shapiro_pval,
            'JB_Statistic': jb_stat,
            'JB_P_Value': jb_pval,
            'Is_Normal': 'Yes' if is_normal else 'No'
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    logger.info("\n" + "=" * 100)
    logger.info("DISTRIBUTION TESTING RESULTS - SUMMARY TABLE")
    logger.info("=" * 100)

    from tabulate import tabulate

    # Display full results
    display_df = results_df.copy()
    display_df = display_df.round(4)
    print("\n" + tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    # Summary statistics
    logger.info("\n>>> DISTRIBUTION SUMMARY:")
    dist_counts = results_df['Best_Distribution'].value_counts()
    for dist, count in dist_counts.items():
        logger.info(f"    {dist}: {count} variables ({count/len(results_df)*100:.1f}%)")

    normal_count = (results_df['Is_Normal'] == 'Yes').sum()
    logger.info(f"\n    Variables passing normality tests: {normal_count}/{len(results_df)} ({normal_count/len(results_df)*100:.1f}%)")

    logger.info("\n✓ Distribution testing complete!")

    return results_df


def analyze_advanced_dependencies(df: pd.DataFrame) -> tuple:
    """
    Advanced dependency analysis using entropy, mutual information, and copulas.

    Returns:
        - entropy_results: DataFrame with entropy measures
        - mi_matrix: Mutual information matrix
        - copula_results: DataFrame with copula dependency measures
    """
    from scipy.stats import entropy as scipy_entropy
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import spearmanr, kendalltau
    import warnings
    warnings.filterwarnings('ignore')

    logger.info("\n" + "=" * 100)
    logger.info("ADVANCED DEPENDENCY ANALYSIS")
    logger.info("Entropy | Mutual Information | Copula Analysis")
    logger.info("=" * 100)

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove columns with too many missing values
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > 100]

    logger.info(f"\n>>> Analyzing {len(valid_cols)} numeric variables")

    # =========================================================================
    # STEP 1: ENTROPY ANALYSIS
    # =========================================================================
    logger.info("\n>>> STEP 1: ENTROPY ANALYSIS (Information Content)")
    logger.info("    Measuring uncertainty/randomness in each variable...")

    entropy_results = []

    for col in valid_cols:
        data = df[col].dropna()

        if len(data) < 2:
            continue

        # Discretize continuous data for entropy calculation
        # Use Freedman-Diaconis rule for bin width
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1/3))
        n_bins = int(np.ceil((data.max() - data.min()) / bin_width)) if bin_width > 0 else 10
        n_bins = min(max(n_bins, 5), 50)  # Clamp between 5 and 50

        # Create histogram
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        hist = hist / hist.sum()  # Normalize to probability distribution

        # Calculate entropy (in bits)
        entr = scipy_entropy(hist, base=2)

        # Calculate normalized entropy (0 to 1)
        max_entropy = np.log2(n_bins)
        normalized_entropy = entr / max_entropy if max_entropy > 0 else 0

        # Calculate coefficient of variation
        cv = data.std() / data.mean() if data.mean() != 0 else np.inf

        logger.info(f"    {col}:")
        logger.info(f"        Entropy: {entr:.4f} bits (normalized: {normalized_entropy:.4f})")
        logger.info(f"        CV: {cv:.4f}")
        logger.info(f"        Interpretation: {'High variability' if normalized_entropy > 0.7 else 'Moderate variability' if normalized_entropy > 0.4 else 'Low variability'}")

        entropy_results.append({
            'Variable': col,
            'Entropy_Bits': entr,
            'Normalized_Entropy': normalized_entropy,
            'Coefficient_of_Variation': cv,
            'N_Bins': n_bins,
            'Variability': 'High' if normalized_entropy > 0.7 else 'Moderate' if normalized_entropy > 0.4 else 'Low'
        })

    entropy_df = pd.DataFrame(entropy_results)

    logger.info("\n    ENTROPY RESULTS TABLE:")
    from tabulate import tabulate
    print(tabulate(entropy_df.round(4), headers='keys', tablefmt='grid', showindex=False))

    # =========================================================================
    # STEP 2: MUTUAL INFORMATION ANALYSIS
    # =========================================================================
    logger.info("\n>>> STEP 2: MUTUAL INFORMATION ANALYSIS (Variable Dependencies)")
    logger.info("    Computing pairwise mutual information between variables...")

    # Prepare data for MI calculation
    df_valid = df[valid_cols].dropna()

    if len(df_valid) < 50:
        logger.warning("    Insufficient data for mutual information analysis")
        mi_matrix = pd.DataFrame()
    else:
        logger.info(f"    Using {len(df_valid):,} complete observations")

        # Compute mutual information matrix
        mi_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols, dtype=float)

        for i, col1 in enumerate(valid_cols):
            for j, col2 in enumerate(valid_cols):
                if i == j:
                    # Self-MI equals entropy
                    mi_matrix.loc[col1, col2] = entropy_df[entropy_df['Variable'] == col1]['Entropy_Bits'].values[0]
                elif i < j:
                    # Compute MI
                    X = df_valid[[col1]].values
                    y = df_valid[col2].values

                    try:
                        mi_value = mutual_info_regression(X, y, random_state=42)[0]
                        mi_matrix.loc[col1, col2] = mi_value
                        mi_matrix.loc[col2, col1] = mi_value

                        if mi_value > 1.0:
                            logger.info(f"    Strong dependency: {col1} ↔ {col2} (MI={mi_value:.4f})")
                    except:
                        mi_matrix.loc[col1, col2] = np.nan
                        mi_matrix.loc[col2, col1] = np.nan

        # Convert to numeric
        mi_matrix = mi_matrix.astype(float)

        # Display top dependencies
        logger.info("\n    TOP 10 MUTUAL INFORMATION PAIRS:")
        mi_pairs = []
        for i in range(len(valid_cols)):
            for j in range(i+1, len(valid_cols)):
                val = mi_matrix.iloc[i, j]
                if not np.isnan(val):
                    mi_pairs.append({
                        'Var1': valid_cols[i],
                        'Var2': valid_cols[j],
                        'MI_Score': val
                    })

        mi_pairs_df = pd.DataFrame(mi_pairs).sort_values('MI_Score', ascending=False).head(10)
        print(tabulate(mi_pairs_df.round(4), headers='keys', tablefmt='grid', showindex=False))

    # =========================================================================
    # STEP 3: COPULA ANALYSIS (Tail Dependencies)
    # =========================================================================
    logger.info("\n>>> STEP 3: COPULA ANALYSIS (Non-linear Dependencies)")
    logger.info("    Computing rank correlations for copula-based dependency structure...")

    copula_results = []

    # Select key variables for copula analysis (avoid computation explosion)
    key_vars = ['PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'MIN', 'TOV', 'STL', 'BLK']
    key_vars = [v for v in key_vars if v in valid_cols]

    logger.info(f"    Analyzing {len(key_vars)} key variables")

    df_copula = df[key_vars].dropna()

    if len(df_copula) < 50:
        logger.warning("    Insufficient data for copula analysis")
        copula_df = pd.DataFrame()
    else:
        logger.info(f"    Using {len(df_copula):,} complete observations")

        for i, var1 in enumerate(key_vars):
            for j, var2 in enumerate(key_vars):
                if i >= j:
                    continue

                x = df_copula[var1].values
                y = df_copula[var2].values

                # Compute rank correlations
                spearman_rho, spearman_p = spearmanr(x, y)
                kendall_tau, kendall_p = kendalltau(x, y)

                # Compute Pearson for comparison
                pearson_r = np.corrcoef(x, y)[0, 1]

                # Tail dependency index (simplified)
                # Upper tail: correlation in top 10%
                q90 = np.percentile(x, 90)
                upper_tail_mask = x >= q90
                if upper_tail_mask.sum() > 10:
                    upper_tail_corr = np.corrcoef(x[upper_tail_mask], y[upper_tail_mask])[0, 1]
                else:
                    upper_tail_corr = np.nan

                # Lower tail: correlation in bottom 10%
                q10 = np.percentile(x, 10)
                lower_tail_mask = x <= q10
                if lower_tail_mask.sum() > 10:
                    lower_tail_corr = np.corrcoef(x[lower_tail_mask], y[lower_tail_mask])[0, 1]
                else:
                    lower_tail_corr = np.nan

                # Determine dependency type
                if abs(spearman_rho - pearson_r) < 0.1:
                    dep_type = "Linear"
                elif abs(spearman_rho) > abs(pearson_r) + 0.1:
                    dep_type = "Monotonic (Non-linear)"
                else:
                    dep_type = "Non-monotonic"

                if abs(spearman_rho) > 0.5 or abs(kendall_tau) > 0.3:
                    logger.info(f"    {var1} ↔ {var2}:")
                    logger.info(f"        Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4f})")
                    logger.info(f"        Kendall τ: {kendall_tau:.4f} (p={kendall_p:.4f})")
                    logger.info(f"        Pearson r: {pearson_r:.4f}")
                    logger.info(f"        Dependency type: {dep_type}")
                    if not np.isnan(upper_tail_corr):
                        logger.info(f"        Upper tail dependency: {upper_tail_corr:.4f}")
                    if not np.isnan(lower_tail_corr):
                        logger.info(f"        Lower tail dependency: {lower_tail_corr:.4f}")

                copula_results.append({
                    'Var1': var1,
                    'Var2': var2,
                    'Pearson_r': pearson_r,
                    'Spearman_ρ': spearman_rho,
                    'Spearman_p': spearman_p,
                    'Kendall_τ': kendall_tau,
                    'Kendall_p': kendall_p,
                    'Upper_Tail_Dep': upper_tail_corr,
                    'Lower_Tail_Dep': lower_tail_corr,
                    'Dependency_Type': dep_type,
                    'Strength': 'Strong' if abs(spearman_rho) > 0.5 else 'Moderate' if abs(spearman_rho) > 0.3 else 'Weak'
                })

        copula_df = pd.DataFrame(copula_results).sort_values('Spearman_ρ', key=abs, ascending=False)

        logger.info("\n    COPULA DEPENDENCY RESULTS - TOP 15 PAIRS:")
        display_copula = copula_df.head(15).copy()
        display_copula = display_copula.round(4)
        print(tabulate(display_copula, headers='keys', tablefmt='grid', showindex=False))

        # Summary by dependency type
        logger.info("\n    DEPENDENCY TYPE SUMMARY:")
        dep_type_counts = copula_df['Dependency_Type'].value_counts()
        for dep_type, count in dep_type_counts.items():
            logger.info(f"        {dep_type}: {count} pairs ({count/len(copula_df)*100:.1f}%)")

        # Summary by strength
        logger.info("\n    DEPENDENCY STRENGTH SUMMARY:")
        strength_counts = copula_df['Strength'].value_counts()
        for strength, count in strength_counts.items():
            logger.info(f"        {strength}: {count} pairs ({count/len(copula_df)*100:.1f}%)")

    logger.info("\n✓ Advanced dependency analysis complete!")
    logger.info("=" * 100)

    return entropy_df, mi_matrix, copula_df


def create_enhanced_seaborn_distributions(df: pd.DataFrame, output_dir: Path, batch_size: int = 4):
    """
    Create enhanced Seaborn distribution plots with KDE curves and annotations.
    Plots are generated in small batches for clarity.

    Args:
        df: DataFrame with game data
        batch_size: Number of plots per figure (default 4 for 2x2 grid)
        output_dir: Directory to save figures

    Returns:
        List of figure file paths
    """
    logger.info("\n" + "=" * 100)
    logger.info("ENHANCED SEABORN DISTRIBUTION PLOTS")
    logger.info("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Key variables to plot
    variables = {
        'Volume Stats': ['PTS', 'AST', 'REB', 'FG3M'],
        'Shooting Stats': ['FGA', 'FG3A', 'FTA', 'MIN'],
        'Shooting %': ['FG_PCT', 'FG3_PCT', 'FT_PCT'],
        'Other Stats': ['BLK', 'STL', 'TOV', 'PLUS_MINUS']
    }

    for category, cols in variables.items():
        # Filter to available columns
        available_cols = [c for c in cols if c in df.columns]

        if not available_cols:
            continue

        # Create batches
        for batch_num in range(0, len(available_cols), batch_size):
            batch_cols = available_cols[batch_num:batch_num + batch_size]

            n_plots = len(batch_cols)
            n_rows = 2
            n_cols = 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
            axes = axes.flatten()

            logger.info(f"\n>>> Creating {category} plots - Batch {batch_num//batch_size + 1}")

            for idx, col in enumerate(batch_cols):
                ax = axes[idx]
                data = df[col].dropna()

                if len(data) < 2:
                    continue

                # Plot histogram with KDE
                sns.histplot(data, kde=True, ax=ax, color='steelblue', alpha=0.6,
                            stat='density', bins=30, edgecolor='black', linewidth=1.2)

                # Enhance KDE line
                kde_line = ax.lines[0] if ax.lines else None
                if kde_line:
                    kde_line.set_color('darkred')
                    kde_line.set_linewidth(2.5)
                    kde_line.set_alpha(0.8)

                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()

                ax.axvline(mean_val, color='green', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.2f}', alpha=0.8)
                ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                          label=f'Median: {median_val:.2f}', alpha=0.8)

                # Add title with stats
                ax.set_title(f'{col} Distribution\n'
                           f'n={len(data):,} | μ={mean_val:.2f} | σ={std_val:.2f}',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel(f'{col} Value', fontsize=10, fontweight='bold')
                ax.set_ylabel('Density', fontsize=10, fontweight='bold')
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3, linestyle=':')

                # Add annotation for key features
                q25, q75 = data.quantile([0.25, 0.75])
                skew = data.skew()

                annotation = f'IQR: [{q25:.2f}, {q75:.2f}]\n'
                if abs(skew) > 1:
                    annotation += f'Highly skewed (γ={skew:.2f})'
                elif abs(skew) > 0.5:
                    annotation += f'Moderately skewed (γ={skew:.2f})'
                else:
                    annotation += f'Approximately symmetric (γ={skew:.2f})'

                ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                logger.info(f"    {col}: Mean={mean_val:.2f}, Median={median_val:.2f}, Skew={skew:.2f}")

            # Hide unused subplots
            for idx in range(n_plots, len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(f'{category} - Batch {batch_num//batch_size + 1}\n'
                        f'Kernel Density Estimation with Histogram',
                        fontsize=16, fontweight='bold', y=0.995)

            plt.tight_layout(rect=[0, 0, 1, 0.99])

            # Save figure
            filename = f'distributions_{category.replace(" ", "_")}_batch{batch_num//batch_size + 1}.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_figures.append(filepath)
            logger.info(f"    Saved: {filename}")

            plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} distribution plot figures")
    return saved_figures


def create_copula_visualizations(df: pd.DataFrame, copula_results: pd.DataFrame, output_dir: Path):
    """
    Create copula dependency visualizations.

    Args:
        df: DataFrame with game data
        copula_results: Results from copula analysis
        output_dir: Directory to save figures

    Returns:
        List of figure file paths
    """
    logger.info("\n" + "=" * 100)
    logger.info("COPULA DEPENDENCY VISUALIZATIONS")
    logger.info("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Get top pairs by absolute Spearman correlation
    top_pairs = copula_results.nlargest(6, 'Spearman_ρ', keep='all')

    # Create 2x3 grid for top 6 pairs
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    logger.info(f"\n>>> Creating scatter plots for top {len(top_pairs)} dependencies")

    for idx, (_, pair) in enumerate(top_pairs.iterrows()):
        if idx >= 6:
            break

        ax = axes[idx]
        var1, var2 = pair['Var1'], pair['Var2']

        # Get data
        plot_data = df[[var1, var2]].dropna()

        if len(plot_data) < 10:
            continue

        # Create scatter with density
        x = plot_data[var1].values
        y = plot_data[var2].values

        # Joint plot
        ax.scatter(x, y, alpha=0.3, s=20, color='steelblue', edgecolor='none')

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_trend, p(x_trend), "r-", linewidth=2, alpha=0.8, label=f'Trend line')

        # Add correlation info
        pearson_r = pair['Pearson_r']
        spearman_rho = pair['Spearman_ρ']
        dep_type = pair['Dependency_Type']

        # Title with correlation info
        ax.set_title(f'{var1} vs {var2}\n'
                    f'ρ_s={spearman_rho:.2f} | r_p={pearson_r:.2f} | {dep_type}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel(var1, fontsize=10, fontweight='bold')
        ax.set_ylabel(var2, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=8)

        # Add tail dependency annotation
        upper_tail = pair.get('Upper_Tail_Dep', np.nan)
        lower_tail = pair.get('Lower_Tail_Dep', np.nan)

        annotation = f'Tail Dependencies:\n'
        if not np.isnan(upper_tail):
            annotation += f'Upper (top 10%): {upper_tail:.2f}\n'
        if not np.isnan(lower_tail):
            annotation += f'Lower (bot 10%): {lower_tail:.2f}'

        ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        logger.info(f"    {var1} ↔ {var2}: ρ={spearman_rho:.2f}, type={dep_type}")

    # Hide unused subplots
    for idx in range(len(top_pairs), 6):
        axes[idx].set_visible(False)

    fig.suptitle('Copula Dependency Analysis\nTop Variable Pairs by Spearman Rank Correlation',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    filename = 'copula_dependencies.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    saved_figures.append(filepath)
    logger.info(f"    Saved: {filename}")

    plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} copula visualization(s)")
    return saved_figures


def create_additional_copula_plots(df: pd.DataFrame, copula_results: pd.DataFrame, output_dir: Path):
    """
    Create additional copula visualizations including rank-transformed plots and heatmaps.

    Args:
        df: DataFrame with game data
        copula_results: Results from copula analysis
        output_dir: Directory to save figures

    Returns:
        List of figure file paths
    """
    logger.info("\n" + "=" * 100)
    logger.info("ADDITIONAL COPULA VISUALIZATIONS")
    logger.info("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Get key variables for copula analysis
    key_vars = ['PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'MIN', 'TOV', 'STL', 'BLK']
    key_vars = [v for v in key_vars if v in df.columns]

    df_copula = df[key_vars].dropna()

    if len(df_copula) < 50:
        logger.warning("Insufficient data for additional copula analysis")
        return saved_figures

    # =========================================================================
    # 1. RANK-TRANSFORMED COPULA SCATTER PLOTS (Actual Copula Structure)
    # =========================================================================
    logger.info("\n>>> Creating rank-transformed copula scatter plots...")

    # Get top 6 pairs
    top_pairs = copula_results.nlargest(6, 'Spearman_ρ', keep='all').head(6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (_, pair) in enumerate(top_pairs.iterrows()):
        if idx >= 6:
            break

        ax = axes[idx]
        var1, var2 = pair['Var1'], pair['Var2']

        # Get data
        plot_data = df[[var1, var2]].dropna()

        if len(plot_data) < 10:
            continue

        x = plot_data[var1].values
        y = plot_data[var2].values

        # Rank transform (this reveals the copula structure)
        from scipy.stats import rankdata
        x_rank = rankdata(x) / (len(x) + 1)  # Convert to uniform [0, 1]
        y_rank = rankdata(y) / (len(y) + 1)

        # Create hexbin plot for density
        hexbin = ax.hexbin(x_rank, y_rank, gridsize=30, cmap='Blues', alpha=0.7,
                          edgecolors='black', linewidths=0.2)

        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label('Count', fontsize=9)

        # Add correlation info
        spearman_rho = pair['Spearman_ρ']
        dep_type = pair['Dependency_Type']

        ax.set_title(f'{var1} vs {var2} (Rank-Transformed)\n'
                    f'Copula Structure | ρ_s={spearman_rho:.2f} | {dep_type}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{var1} Rank (Uniform)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{var2} Rank (Uniform)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')

        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5, label='Independence')
        ax.legend(fontsize=8)

        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        logger.info(f"    Rank-transformed: {var1} ↔ {var2}")

    # Hide unused subplots
    for idx in range(len(top_pairs), 6):
        axes[idx].set_visible(False)

    fig.suptitle('Copula Structure Analysis (Rank-Transformed)\n'
                'Hexbin Density Plots Showing Pure Dependence Structure',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    filename = 'copula_rank_transformed.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    saved_figures.append(filepath)
    logger.info(f"    Saved: {filename}")

    plt.close()

    # =========================================================================
    # 2. COPULA DEPENDENCY HEATMAP
    # =========================================================================
    logger.info("\n>>> Creating copula dependency heatmap...")

    # Create correlation matrix from copula results
    unique_vars = sorted(set(copula_results['Var1'].tolist() + copula_results['Var2'].tolist()))

    corr_matrix = pd.DataFrame(1.0, index=unique_vars, columns=unique_vars)

    for _, row in copula_results.iterrows():
        var1, var2 = row['Var1'], row['Var2']
        rho = row['Spearman_ρ']
        corr_matrix.loc[var1, var2] = rho
        corr_matrix.loc[var2, var1] = rho

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={'label': 'Spearman ρ'}, ax=ax, annot_kws={'fontsize': 9})

    ax.set_title('Copula Dependency Heatmap\nSpearman Rank Correlation Matrix',
                fontsize=16, fontweight='bold')

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    filename = 'copula_heatmap.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    saved_figures.append(filepath)
    logger.info(f"    Saved: {filename}")

    plt.close()

    # =========================================================================
    # 3. TAIL DEPENDENCY COMPARISON PLOT
    # =========================================================================
    logger.info("\n>>> Creating tail dependency comparison plot...")

    # Get pairs with tail dependency data
    tail_pairs = copula_results[
        copula_results['Upper_Tail_Dep'].notna() &
        copula_results['Lower_Tail_Dep'].notna()
    ].nlargest(10, 'Spearman_ρ', keep='all').head(10)

    if len(tail_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Prepare data
        pair_labels = [f"{row['Var1']}-{row['Var2']}" for _, row in tail_pairs.iterrows()]
        upper_tail = tail_pairs['Upper_Tail_Dep'].values
        lower_tail = tail_pairs['Lower_Tail_Dep'].values
        overall_corr = tail_pairs['Spearman_ρ'].values

        y_pos = np.arange(len(pair_labels))

        # Plot 1: Upper vs Lower Tail Dependencies
        ax1.barh(y_pos - 0.2, upper_tail, 0.4, label='Upper Tail (Top 10%)',
                color='red', alpha=0.7, edgecolor='black')
        ax1.barh(y_pos + 0.2, lower_tail, 0.4, label='Lower Tail (Bottom 10%)',
                color='blue', alpha=0.7, edgecolor='black')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pair_labels, fontsize=9)
        ax1.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax1.set_title('Tail Dependencies Comparison\nUpper vs Lower Tail Correlations',
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(0, color='black', linewidth=0.8)

        # Add value labels
        for i, (u, l) in enumerate(zip(upper_tail, lower_tail)):
            ax1.text(u + 0.02, i - 0.2, f'{u:.2f}', va='center', fontsize=8)
            ax1.text(l + 0.02, i + 0.2, f'{l:.2f}', va='center', fontsize=8)

        # Plot 2: Tail Asymmetry
        asymmetry = upper_tail - lower_tail

        colors = ['green' if a > 0 else 'orange' for a in asymmetry]
        ax2.barh(y_pos, asymmetry, color=colors, alpha=0.7, edgecolor='black')

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pair_labels, fontsize=9)
        ax2.set_xlabel('Tail Asymmetry (Upper - Lower)', fontsize=11, fontweight='bold')
        ax2.set_title('Tail Dependency Asymmetry\nPositive = Stronger Upper Tail',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(0, color='black', linewidth=2)

        # Add value labels
        for i, a in enumerate(asymmetry):
            x_pos = a + (0.02 if a > 0 else -0.02)
            ha = 'left' if a > 0 else 'right'
            ax2.text(x_pos, i, f'{a:.2f}', va='center', ha=ha, fontsize=8, fontweight='bold')

        fig.suptitle('Copula Tail Dependency Analysis\n'
                    'Extreme Value Correlations (Top 10% vs Bottom 10%)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        filename = 'copula_tail_dependencies.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_figures.append(filepath)
        logger.info(f"    Saved: {filename}")

        plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} additional copula visualization(s)")
    return saved_figures


def create_team_consistency_plots(df: pd.DataFrame, prop_analysis: pd.DataFrame, output_dir: Path, teams_per_page: int = 3):
    """
    Create team-by-team consistency plots in small batches.

    Args:
        df: Game-level data
        prop_analysis: Player consistency metrics
        output_dir: Directory to save figures
        teams_per_page: Number of teams per figure

    Returns:
        List of figure file paths
    """
    logger.info("\n" + "=" * 100)
    logger.info("TEAM CONSISTENCY VISUALIZATIONS")
    logger.info("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Get all teams
    teams = sorted(df['player_team'].unique())
    total_teams = len(teams)
    total_pages = (total_teams + teams_per_page - 1) // teams_per_page

    logger.info(f"\n>>> Creating {total_pages} figures for {total_teams} teams ({teams_per_page} teams per figure)")

    for page_num in range(total_pages):
        start_idx = page_num * teams_per_page
        end_idx = min((page_num + 1) * teams_per_page, total_teams)
        teams_on_page = teams[start_idx:end_idx]

        n_teams = len(teams_on_page)
        fig, axes = plt.subplots(n_teams, 1, figsize=(14, 5 * n_teams))

        if n_teams == 1:
            axes = [axes]

        logger.info(f"\n>>> Page {page_num + 1}/{total_pages}: Teams {start_idx + 1}-{end_idx}")

        for team_idx, team in enumerate(teams_on_page):
            ax = axes[team_idx]

            # Get team players
            team_players = df[df['player_team'] == team]['player_name'].unique()

            # Get consistency data
            team_consistency = prop_analysis[prop_analysis['player_name'].isin(team_players)]
            team_consistency = team_consistency.sort_values('PPG', ascending=False).head(15)

            if len(team_consistency) == 0:
                ax.text(0.5, 0.5, f'No data for {team}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(team, fontweight='bold', fontsize=14)
                continue

            # Create scatter plot: Consistency vs Performance
            x = team_consistency['PPG'].values
            y = team_consistency['PTS_CV'].values

            # Color by consistency (green = consistent, red = inconsistent)
            colors = ['green' if cv < 0.40 else 'orange' if cv < 0.60 else 'red' for cv in y]

            ax.scatter(x, y, s=150, c=colors, alpha=0.6, edgecolor='black', linewidth=1.5)

            # Add player labels for top 5
            for i in range(min(5, len(team_consistency))):
                player = team_consistency.iloc[i]
                ax.annotate(player['player_name'].split()[-1],  # Last name only
                          (player['PPG'], player['PTS_CV']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, fontweight='bold')

            # Add consistency threshold line
            ax.axhline(0.40, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                      label='Consistency threshold (CV=0.40)')

            ax.set_title(f'{team} - Player Consistency vs Performance\n'
                        f'Top {len(team_consistency)} Players',
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Points Per Game (PPG)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Coefficient of Variation (CV)', fontsize=10, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':')

            # Annotate regions
            ax.text(0.98, 0.98, 'Inconsistent\nHigh-scorers', transform=ax.transAxes,
                   ha='right', va='top', fontsize=8, style='italic',
                   bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.3))
            ax.text(0.98, 0.02, 'Consistent\nHigh-scorers', transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=8, style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

            logger.info(f"    {team}: {len(team_consistency)} players plotted")

        fig.suptitle(f'Team Consistency Analysis - Page {page_num + 1}/{total_pages}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        filename = f'team_consistency_page{page_num + 1}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        saved_figures.append(filepath)
        logger.info(f"    Saved: {filename}")

        plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} team consistency figures")
    return saved_figures


def generate_comprehensive_pdf(output_dir: Path, all_figures: list, log_file: Path = None):
    """
    Generate comprehensive PDF with all figures and logs.

    Args:
        output_dir: Directory containing figures
        all_figures: List of all figure paths
        log_file: Optional path to log file

    Returns:
        Path to generated PDF
    """
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    logger.info("\n" + "=" * 100)
    logger.info("GENERATING COMPREHENSIVE PDF REPORT")
    logger.info("=" * 100)

    # Generate filename with current date/time
    now = datetime.now()
    pdf_filename = f"{now.strftime('%H:%M:%Y%m%d')}.pdf"
    pdf_path = output_dir / pdf_filename

    logger.info(f"\n>>> Creating PDF: {pdf_filename}")
    logger.info(f"    Total figures to include: {len(all_figures)}")

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = f"""
NBA PLAYER STATISTICS
COMPREHENSIVE ANALYSIS REPORT

2024-2025 Season

Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}

Total Figures: {len(all_figures)}

Analysis Includes:
• Distribution Analysis with KDE
• Advanced Dependency Analysis (Entropy, MI, Copulas)
• Team Consistency Metrics
• Accumulator Backtesting Simulations
• Statistical Goodness-of-Fit Testing

All figures rounded to 2 decimal places
        """

        ax.text(0.5, 0.5, title_text, ha='center', va='center',
               fontsize=14, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        logger.info("    Added title page")

        # Add all figures
        for idx, fig_path in enumerate(all_figures, 1):
            if not fig_path.exists():
                logger.warning(f"    Figure not found: {fig_path}")
                continue

            try:
                img = mpimg.imread(fig_path)
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')

                # Add caption
                caption = fig_path.stem.replace('_', ' ').title()
                fig.text(0.5, 0.02, f'{idx}/{len(all_figures)}: {caption}',
                        ha='center', fontsize=10, style='italic')

                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                if idx % 10 == 0:
                    logger.info(f"    Added {idx}/{len(all_figures)} figures...")

            except Exception as e:
                logger.warning(f"    Could not add figure {fig_path}: {e}")
                continue

        logger.info(f"    Added all {len(all_figures)} figures")

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'NBA Player Statistics Analysis Report'
        d['Author'] = 'NBA Analytics System'
        d['Subject'] = '2024-2025 Season Analysis'
        d['Keywords'] = 'NBA, Statistics, Analysis, Machine Learning'
        d['CreationDate'] = now

    logger.info(f"\n✓ PDF generated successfully: {pdf_path}")
    logger.info(f"    File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("=" * 100)

    return pdf_path


def main():
    """Main execution function."""
    logger.info("\n" + "=" * 100)
    logger.info("NBA PLAYER STATISTICS COMPREHENSIVE ANALYSIS")
    logger.info("2024-2025 SEASON")
    logger.info("=" * 100)

    # File path
    data_file = Path("/Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research/data/raw/PlayerStatistics_2024_2025.csv")

    # Step 1: Load and display raw data
    df_raw = load_and_display_raw_data(data_file)

    # Step 2: Clean and prepare
    df_clean = clean_and_prepare_data(df_raw)

    # Step 3: Player-level analysis
    analyze_player_level_stats(df_clean)

    # Step 4: Aggregate by player (season averages)
    player_agg = aggregate_by_player(df_clean)

    # Step 5: Aggregate by team
    team_agg = aggregate_by_team(df_clean)

    # Step 6: Home vs away analysis
    home_away_stats = analyze_home_away_performance(df_clean)

    # Step 7: Win vs loss analysis
    win_loss_stats = analyze_win_loss_performance(df_clean)

    # Step 8: Temporal trends
    daily_stats = analyze_temporal_trends(df_clean)

    # Step 9: Prop bet opportunities
    prop_analysis = identify_prop_bet_opportunities(df_clean, player_agg)

    # Step 9a: Line Hit Rate Analysis
    # Analyzes success rates for props at different line positions (avg, avg±5, avg±10)
    # Shows % of games players go OVER each line - critical for line shopping
    line_hit_rates = analyze_line_hit_rates(df_clean, prop_analysis)

    # Step 10: Accumulator backtesting - Date Range Simulation
    # Test your acca structure across all available dates
    # To customize: Edit the acca_template in backtest_acca_across_dates()
    # To change date range: Pass start_date='2024-11-01', end_date='2024-11-30'
    daily_results, all_props = backtest_acca_across_dates(
        df_clean,
        prop_analysis,
        start_date=None,  # None = use all available data
        end_date=None
    )

    # Step 11: Distribution Testing - Identify statistical distributions
    # Tests each variable against 8 distributions (Normal, Lognormal, Gamma, etc.)
    # Uses Kolmogorov-Smirnov, Anderson-Darling, Shapiro-Wilk, and Jarque-Bera tests
    distribution_results = test_distributions(df_clean)

    # Step 12: Advanced Dependency Analysis
    # Entropy, Mutual Information, and Copula analysis
    # Identifies non-linear dependencies and tail dependencies
    entropy_results, mi_matrix, copula_results = analyze_advanced_dependencies(df_clean)

    # Step 13: Create Enhanced Visualizations
    # Enhanced Seaborn distributions, copula plots, and team consistency plots
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_figures = []

    # Create enhanced distribution plots (batched)
    dist_figures = create_enhanced_seaborn_distributions(df_clean, output_dir, batch_size=4)
    all_figures.extend(dist_figures)

    # Create copula visualizations
    if len(copula_results) > 0:
        copula_figures = create_copula_visualizations(df_clean, copula_results, output_dir)
        all_figures.extend(copula_figures)

        # Create additional copula plots (rank-transformed, heatmaps, tail dependencies)
        additional_copula_figures = create_additional_copula_plots(df_clean, copula_results, output_dir)
        all_figures.extend(additional_copula_figures)

    # Create team consistency plots (batched by team)
    team_figures = create_team_consistency_plots(df_clean, prop_analysis, output_dir, teams_per_page=3)
    all_figures.extend(team_figures)

    # Step 14: Generate Comprehensive PDF Report
    # Combine all figures and logs into a single timestamped PDF
    pdf_path = generate_comprehensive_pdf(output_dir, all_figures)

    # Final Summary
    logger.info("\n" + "=" * 100)
    logger.info("ANALYSIS COMPLETE - FINAL SUMMARY")
    logger.info("=" * 100)

    summary_stats = {
        'Metric': [
            'Total Games Analyzed',
            'Total Unique Players',
            'Total Unique Teams',
            'Date Range',
            'Average Points Per Game',
            'Average Assists Per Game',
            'Average Rebounds Per Game',
            'Overall Win Rate',
            'Home Win Rate',
            'Away Win Rate'
        ],
        'Value': [
            f"{df_clean['gameId'].nunique():,}",
            f"{df_clean['player_name'].nunique():,}",
            f"{df_clean['player_team'].nunique():,}",
            f"{df_clean['game_date_only'].min()} to {df_clean['game_date_only'].max()}",
            f"{df_clean['PTS'].mean():.2f}",
            f"{df_clean['AST'].mean():.2f}",
            f"{df_clean['REB'].mean():.2f}",
            f"{df_clean['win'].mean()*100:.1f}%",
            f"{df_clean[df_clean['home']==True]['win'].mean()*100:.1f}%",
            f"{df_clean[df_clean['home']==False]['win'].mean()*100:.1f}%"
        ]
    }

    summary_df = pd.DataFrame(summary_stats)
    print("\n" + tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
    print()

    logger.info("\n✓ ALL ANALYSIS COMPLETE!")
    logger.info("=" * 100)

    print(f"\n{'='*100}")
    print("📄 COMPREHENSIVE PDF REPORT GENERATED")
    print(f"{'='*100}")
    print(f"\nPDF Location: {pdf_path}")
    print(f"Total Figures: {len(all_figures)}")
    print(f"File Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\nFilename Format: HH:MM:YYYYMMDD.pdf")
    print(f"Generated: {pdf_path.name}")
    print(f"{'='*100}\n")

    return df_clean, player_agg, team_agg, prop_analysis, line_hit_rates, daily_results, all_props, distribution_results, entropy_results, mi_matrix, copula_results, pdf_path


if __name__ == "__main__":
    df_clean, player_agg, team_agg, prop_analysis, line_hit_rates, daily_results, all_props, distribution_results, entropy_results, mi_matrix, copula_results, pdf_path = main()

    # Keep variables in scope for interactive use
    print("\n>>> DataFrames available for further analysis:")
    print("    - df_clean: Cleaned game-level data")
    print("    - player_agg: Player season averages")
    print("    - team_agg: Team aggregated stats")
    print("    - prop_analysis: Prop bet opportunity analysis")
    print("    - line_hit_rates: Hit rates at different line positions (by team)")
    print("    - daily_results: Accumulator results by date")
    print("    - all_props: Individual prop results across all dates")
    print("    - distribution_results: Statistical distribution testing results")
    print("    - entropy_results: Entropy analysis results")
    print("    - mi_matrix: Mutual information matrix")
    print("    - copula_results: Copula dependency analysis results")
    print("    - pdf_path: Path to generated PDF report")

    # Show how to run again
    print("\n" + "=" * 100)
    print("TO RUN THIS ANALYSIS AGAIN:")
    print("=" * 100)
    print("\nOption 1 - Direct Python:")
    print("    cd /Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research")
    print("    python data/dataedits.py")
    print("\nOption 2 - Shell Script:")
    print("    cd /Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research")
    print("    ./run_nba_analysis.sh")
    print("\nOption 3 - Interactive Mode:")
    print("    cd /Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research")
    print("    python -i data/dataedits.py")
    print("    # Then explore: df_clean.head(), player_agg.describe(), etc.")
    print("\n" + "=" * 100)

    # Show how to run simulations
    print("\n" + "=" * 100)
    print("🎯 HOW TO RUN ACCUMULATOR SIMULATIONS:")
    print("=" * 100)
    print("\n📍 METHOD 1: Simplified Simulator (RECOMMENDED - Just props + number of games)")
    print("    >>> results, props = simulate_acca(df_clean, prop_analysis)")
    print("\n    This will interactively ask you for:")
    print("        1. Player names (with search and validation)")
    print("        2. Stat type (PTS/AST/REB/BLK/STL/FG3M/TOV/MIN)")
    print("        3. Line value")
    print("        4. Over or Under")
    print("        5. Number of games to simulate")
    print("\n    Or provide props programmatically:")
    print("    >>> props = [")
    print("            {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 20.5, 'over': True},")
    print("            {'player': 'Mikal Bridges', 'stat': 'PTS', 'line': 18.5, 'over': True}")
    print("        ]")
    print("    >>> results, props_df = simulate_acca(df_clean, prop_analysis, props=props, n_games=10)")
    print("\n    Returns:")
    print("        - results: Game-by-game results (wins/losses, hit rates)")
    print("        - props_df: Individual prop performance across all simulated games")
    print("\n📍 METHOD 2: Advanced Builder (with legs structure)")
    print("    >>> daily_df, props_df = interactive_acca_builder(df_clean, prop_analysis)")
    print("\n📍 METHOD 3: Edit Template Directly (for repeated testing)")
    print("    File: data/dataedits.py")
    print("    Location: Line ~1900 in backtest_acca_across_dates() function")
    print("    Look for: # 🎯 CUSTOMIZE YOUR ACCA HERE")
    print("\n    Example structure:")
    print("    acca_template = {")
    print("        'name': 'My Custom Acca',")
    print("        'legs': [")
    print("            {")
    print("                'name': 'Leg 1',")
    print("                'props': [")
    print("                    {'player_last_name': 'Ball', 'stat': 'PTS', 'line': 25.5, 'over': True},")
    print("                    {'player_last_name': 'Curry', 'stat': 'FG3M', 'line': 4.5, 'over': True},")
    print("                ]")
    print("            },")
    print("            # Add more legs...")
    print("        ]")
    print("    }")
    print("\n    Stats available: PTS, AST, REB, BLK, STL, FG3M, FGM, FTM, TOV, MIN")
    print("    Date format: 'YYYY-MM-DD'")
    print("\n📍 METHOD 2: Use Interactive Builder (for quick tests)")
    print("    Run this in Python:")
    print("    >>> daily_df, props_df = interactive_acca_builder(df_clean, prop_analysis)")
    print("\n    The builder will:")
    print("    ✓ Let you search for players by name")
    print("    ✓ Show season averages for each player")
    print("    ✓ Validate all inputs")
    print("    ✓ Confirm your legs before running")
    print("    ✓ Run full backtest with visualizations")
    print("\n📍 METHOD 3: Call Backtest Function Directly")
    print("    After editing the template in METHOD 1:")
    print("    >>> daily_df, props_df = backtest_acca_across_dates(")
    print("            df_clean, prop_analysis,")
    print("            start_date='2024-11-01',  # Optional")
    print("            end_date='2024-11-30'     # Optional")
    print("        )")
    print("\n" + "=" * 100)
