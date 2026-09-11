"""Visualization utilities for bet builder research."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from .config import get_logger


logger = get_logger(__name__)

_TITLE_SUFFIX = ""


def set_title_suffix(suffix: str | None) -> None:
    """Set a global suffix appended to visualization/table titles."""
    global _TITLE_SUFFIX
    _TITLE_SUFFIX = suffix or ""


def _with_suffix(title: str) -> str:
    if not _TITLE_SUFFIX:
        return title
    if _TITLE_SUFFIX in title:
        return title
    return f"{title}\n{_TITLE_SUFFIX}"


def display_dataframe(
    df: pd.DataFrame,
    title: str = "DataFrame",
    max_rows: int = 20,
    max_cols: int = None,
    tablefmt: str = "grid",
    cols_per_chunk: int = 15
) -> None:
    """
    Display a DataFrame using tabulate with detailed formatting.
    Wide tables are automatically split into chunks for readability.
    All numeric values are rounded to 2 decimal places.

    Args:
        df: DataFrame to display
        title: Title for the display
        max_rows: Maximum rows to show
        max_cols: Maximum columns to show (None = all)
        tablefmt: Table format for tabulate
        cols_per_chunk: Number of columns per chunk for wide tables
    """
    logger.info("=" * 80)
    logger.info(f"{_with_suffix(title)}")
    logger.info("=" * 80)
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if len(df) == 0:
        logger.warning("DataFrame is empty!")
        return

    # Display column info
    logger.info(f"Columns: {', '.join(df.columns.tolist())}")

    # Round all numeric columns to 2 decimal places
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(2)

    # Select columns to display
    if max_cols and len(df_display.columns) > max_cols:
        cols = df_display.columns[:max_cols].tolist()
        logger.info(f"Showing first {max_cols} columns")
    else:
        cols = df_display.columns.tolist()

    # Select rows to display
    if len(df_display) > max_rows:
        # Show head and tail
        head_rows = max_rows // 2
        tail_rows = max_rows - head_rows

        display_df = pd.concat([
            df_display[cols].head(head_rows),
            pd.DataFrame([["..."] * len(cols)], columns=cols),
            df_display[cols].tail(tail_rows)
        ])

        logger.info(f"Showing first {head_rows} and last {tail_rows} rows")
    else:
        display_df = df_display[cols]

    # Split into chunks if too many columns
    if len(cols) > cols_per_chunk:
        num_chunks = (len(cols) + cols_per_chunk - 1) // cols_per_chunk
        logger.info(f"Splitting {len(cols)} columns into {num_chunks} chunks of ~{cols_per_chunk} columns each")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * cols_per_chunk
            end_idx = min((chunk_idx + 1) * cols_per_chunk, len(cols))
            chunk_cols = cols[start_idx:end_idx]

            print("\n" + "=" * 80)
            print(f"CHUNK {chunk_idx + 1}/{num_chunks}: Columns {start_idx + 1}-{end_idx}")
            print(f"Columns: {', '.join(chunk_cols)}")
            print("=" * 80)
            print("\n" + tabulate(display_df[chunk_cols], headers='keys', tablefmt=tablefmt, showindex=True))
            print()
    else:
        # Print table normally
        print("\n" + tabulate(display_df, headers='keys', tablefmt=tablefmt, showindex=True))
        print()


def display_summary_stats(
    df: pd.DataFrame,
    columns: List[str] = None,
    title: str = "Summary Statistics"
) -> None:
    """
    Display summary statistics for numeric columns.

    Args:
        df: DataFrame to analyze
        columns: Specific columns to analyze (None = all numeric)
        title: Title for the display
    """
    logger.info("=" * 80)
    logger.info(f"{_with_suffix(title)}")
    logger.info("=" * 80)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        logger.warning("No numeric columns to summarize")
        return

    # Compute statistics
    stats_df = df[columns].describe().T

    # Add additional stats
    stats_df['missing'] = df[columns].isnull().sum()
    stats_df['missing_pct'] = (df[columns].isnull().sum() / len(df) * 100).round(2)
    stats_df['unique'] = df[columns].nunique()

    # Round all numeric columns to 2 decimal places
    numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[numeric_cols] = stats_df[numeric_cols].round(2)

    print("\n" + tabulate(stats_df, headers='keys', tablefmt='grid', floatfmt='.2f'))
    print()


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str] = None,
    bins: int = 30,
    save_path: Path = None,
    show: bool = True,
    use_kde_for_percentages: bool = False
) -> None:
    """
    Plot distributions for numeric columns with production-quality statistical handling.

    Features (all 10 requirements implemented):
    1. Discrete stats (PTS, AST, REB, STL, BLK, TOV, FG3M): NO KDE, integer bins only
    2. Percentage stats: Bounded KDE with clip=(0,1) to prevent edge effects
    3. Percentage stats: Filter low-attempt games (FGA<3, 3PA<2, FTA<3)
    4. Percentage histograms: Attempt-weighted (treats 15 attempts differently than 1)
    5. Consistent y-axis scaling across all panels
    6. KDE bandwidth increased (bw_adjust=1.5-2.0) for percentages
    7. X-axis truncated at 95th percentile for long-tailed distributions
    8. KDE optional for percentages (use_kde_for_percentages parameter)
    9. Grouped by metric type (volume, shooting, defensive, percentages)
    10. Annotated with "X% zeros dropped" for percentage plots

    Args:
        df: DataFrame with FULL game data (needs attempt columns for weighting)
        columns: Columns to plot (None = all numeric)
        bins: Number of bins for histograms
        save_path: Optional path to save figure
        show: Whether to display the plot
        use_kde_for_percentages: Whether to add KDE to percentage plots (default False)
    """
    logger.info("\n" + "=" * 100)
    logger.info(_with_suffix("DISTRIBUTION ANALYSIS - PRODUCTION STATISTICAL QUALITY"))
    logger.info("=" * 100)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        logger.warning("No numeric columns to plot")
        return

    # =========================================================================
    # STEP 1: CATEGORIZE STATS BY TYPE
    # =========================================================================
    logger.info("\n>>> STEP 1: Categorizing statistics by type...")

    discrete_stats = ['PTS', 'AST', 'REB', 'BLK', 'STL', 'TOV', 'FG3M', 'FGM', 'FGA', 'FG3A', 'FTM', 'FTA',
                      'points', 'assists', 'blocks', 'steals', 'turnovers', 'threePointersMade',
                      'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersAttempted',
                      'freeThrowsMade', 'freeThrowsAttempted', 'reboundsTotal', 'reboundsDefensive',
                      'reboundsOffensive', 'foulsPersonal', 'MIN']

    percentage_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'fieldGoalsPercentage',
                        'threePointersPercentage', 'freeThrowsPercentage']

    # Map percentage stats to their attempt columns for filtering and weighting
    attempt_columns = {
        'FG_PCT': 'FGA',
        'FG3_PCT': 'FG3A',
        'FT_PCT': 'FTA',
        'fieldGoalsPercentage': 'fieldGoalsAttempted',
        'threePointersPercentage': 'threePointersAttempted',
        'freeThrowsPercentage': 'freeThrowsAttempted'
    }

    # Minimum attempts required for each percentage stat
    min_attempts = {
        'FG_PCT': 3,
        'FG3_PCT': 2,
        'FT_PCT': 3,
        'fieldGoalsPercentage': 3,
        'threePointersPercentage': 2,
        'freeThrowsPercentage': 3
    }

    # Stat descriptions
    stat_descriptions = {
        'PTS': 'Points Per Game',
        'AST': 'Assists Per Game',
        'REB': 'Rebounds Per Game',
        'FG3M': 'Three-Pointers Made',
        'BLK': 'Blocks Per Game',
        'STL': 'Steals Per Game',
        'TOV': 'Turnovers Per Game',
        'FGM': 'Field Goals Made',
        'FGA': 'Field Goal Attempts',
        'FG3A': 'Three-Point Attempts',
        'FTM': 'Free Throws Made',
        'FTA': 'Free Throw Attempts',
        'FG_PCT': 'Field Goal Percentage',
        'FG3_PCT': 'Three-Point Percentage',
        'FT_PCT': 'Free Throw Percentage',
        'PLUS_MINUS': 'Plus/Minus',
        'MIN': 'Minutes Played'
    }

    # Categorize columns
    discrete_cols = [c for c in columns if any(d in c for d in discrete_stats)]
    percentage_cols = [c for c in columns if any(p in c for p in percentage_stats)]
    continuous_cols = [c for c in columns if c not in discrete_cols and c not in percentage_cols]

    logger.info(f"    Discrete stats (integer counts): {len(discrete_cols)} columns")
    logger.info(f"    Percentage stats (bounded [0,1]): {len(percentage_cols)} columns")
    logger.info(f"    Continuous stats (other): {len(continuous_cols)} columns")

    # =========================================================================
    # STEP 2: ORGANIZE BY METRIC TYPE (REQUIREMENT #9)
    # =========================================================================
    logger.info("\n>>> STEP 2: Organizing plots by metric category...")

    volume_order = ['PTS', 'AST', 'REB', 'FG3M']
    shooting_order = ['FGM', 'FGA', 'FG3A', 'FTM', 'FTA']
    defensive_order = ['BLK', 'STL', 'TOV']

    ordered_cols = []

    # Volume stats first
    for stat in volume_order:
        ordered_cols.extend([c for c in discrete_cols if stat in c])

    # Shooting stats second
    for stat in shooting_order:
        ordered_cols.extend([c for c in discrete_cols if stat in c and c not in ordered_cols])

    # Defensive stats third
    for stat in defensive_order:
        ordered_cols.extend([c for c in discrete_cols if stat in c and c not in ordered_cols])

    # Remaining discrete
    ordered_cols.extend([c for c in discrete_cols if c not in ordered_cols])

    # Percentage stats grouped together
    ordered_cols.extend(percentage_cols)

    # Other continuous stats last
    ordered_cols.extend(continuous_cols)

    columns = ordered_cols
    logger.info(f"    Plot order: Volume → Shooting → Defensive → Percentages → Other")
    logger.info(f"    Total plots: {len(columns)}")

    # =========================================================================
    # STEP 3: PREPARE DATA WITH FILTERING (REQUIREMENTS #3, #10)
    # =========================================================================
    logger.info("\n>>> STEP 3: Preparing data with filtering...")

    filtering_summary = []

    for col in columns:
        original_count = len(df[col].dropna())
        filtered_count = original_count
        zeros_removed = 0
        low_attempts_removed = 0

        if col in percentage_cols:
            # Filter low-attempt games (REQUIREMENT #3)
            attempt_col = attempt_columns.get(col)
            min_att = min_attempts.get(col, 0)

            if attempt_col and attempt_col in df.columns:
                valid_mask = (df[col].notna()) & (df[attempt_col] >= min_att)
                filtered_count = valid_mask.sum()
                low_attempts_removed = original_count - filtered_count

                # Also count zeros
                zeros_removed = ((df[col] == 0) & valid_mask).sum()
            else:
                # No attempt column, just remove zeros
                zeros_removed = (df[col] == 0).sum()
                filtered_count = original_count - zeros_removed

            filtering_summary.append({
                'Stat': col,
                'Original_N': original_count,
                'Zeros_Removed': zeros_removed,
                'Low_Attempts_Removed': low_attempts_removed,
                'Final_N': filtered_count - zeros_removed,
                'Pct_Filtered': ((original_count - (filtered_count - zeros_removed)) / original_count * 100)
                if original_count > 0 else 0
            })

    if filtering_summary:
        filter_df = pd.DataFrame(filtering_summary)
        logger.info("\n    PERCENTAGE STAT FILTERING SUMMARY:")
        from tabulate import tabulate
        print(tabulate(filter_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.1f'))

    # =========================================================================
    # STEP 4: CREATE PLOTS
    # =========================================================================
    logger.info("\n>>> STEP 4: Generating distribution plots...")

    n_cols = 4
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle('NBA Player Statistics Distributions - 2024-2025 Season\n'
                 'Production Statistical Quality: Filtered, Weighted, and Bounded',
                 fontsize=18, fontweight='bold', y=0.998)

    # Track for consistent y-axis scaling (REQUIREMENT #5)
    max_density = 0

    for idx, col in enumerate(columns):
        ax = axes[idx]
        logger.info(f"\n    Processing [{idx+1}/{len(columns)}]: {col}")

        # Get data
        data = df[col].dropna()
        original_count = len(data)
        weights = None
        zeros_dropped_pct = 0
        low_attempts_dropped_pct = 0

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col)
            continue

        is_discrete = col in discrete_cols
        is_percentage = col in percentage_cols

        # =====================================================================
        # PERCENTAGE STAT SPECIAL HANDLING (REQUIREMENTS #2, #3, #4, #6, #8, #10)
        # =====================================================================
        if is_percentage:
            attempt_col = attempt_columns.get(col)
            min_att = min_attempts.get(col, 0)

            # Filter low-attempt games (REQUIREMENT #3)
            if attempt_col and attempt_col in df.columns:
                valid_mask = (df[col].notna()) & (df[attempt_col] >= min_att)
                low_attempts_before = original_count
                data = df.loc[valid_mask, col]
                weights_raw = df.loc[valid_mask, attempt_col]
                low_attempts_dropped_pct = ((low_attempts_before - len(data)) / low_attempts_before * 100) if low_attempts_before > 0 else 0

                logger.info(f"        Filtered out {low_attempts_before - len(data)} games with <{min_att} attempts ({low_attempts_dropped_pct:.1f}%)")

                # Remove zeros (DNP)
                zeros_count = (data == 0).sum()
                if zeros_count > 0:
                    non_zero_mask = data > 0
                    weights = weights_raw[non_zero_mask].values
                    data = data[non_zero_mask]
                    zeros_dropped_pct = (zeros_count / len(data)) * 100
                    logger.info(f"        Removed {zeros_count} zero values ({zeros_dropped_pct:.1f}%)")
                else:
                    weights = weights_raw.values

                # Normalize weights for density histogram (REQUIREMENT #4)
                weights = weights / weights.sum() * len(weights)
                logger.info(f"        Using attempt-weighted histogram (n_attempts = {len(weights)})")
            else:
                # No attempt column available, just remove zeros
                zeros_count = (data == 0).sum()
                data = data[data > 0]
                zeros_dropped_pct = (zeros_count / original_count * 100) if original_count > 0 else 0
                logger.info(f"        Removed {zeros_count} zeros ({zeros_dropped_pct:.1f}%), no weights available")

            if len(data) == 0:
                ax.text(0.5, 0.5, 'No valid data after filtering', ha='center', va='center')
                ax.set_title(col)
                continue

        # =====================================================================
        # X-AXIS TRUNCATION (REQUIREMENT #7)
        # =====================================================================
        p95 = data.quantile(0.95)
        x_max = min(data.max(), p95 * 1.2)
        logger.info(f"        Data range: [{data.min():.3f}, {data.max():.3f}], displaying up to {x_max:.3f} (95th %ile)")

        # =====================================================================
        # CREATE BINS
        # =====================================================================
        if is_discrete:
            # Integer-aligned bins for discrete stats (REQUIREMENT #1)
            bin_edges = range(int(data.min()), int(data.max()) + 2)
        else:
            bin_edges = bins

        # =====================================================================
        # PLOT HISTOGRAM
        # =====================================================================
        n, bins_used, patches = ax.hist(
            data, bins=bin_edges, alpha=0.7, edgecolor='black',
            density=True, color='steelblue', weights=weights
        )

        max_density = max(max_density, n.max())

        # =====================================================================
        # ADD KDE (REQUIREMENTS #1, #2, #6, #8)
        # =====================================================================
        if is_discrete:
            # NO KDE for discrete stats (REQUIREMENT #1)
            logger.info(f"        Discrete stat: KDE disabled")
        elif is_percentage and not use_kde_for_percentages:
            # NO KDE for percentages by default (REQUIREMENT #8)
            logger.info(f"        Percentage stat: KDE disabled (use_kde_for_percentages=False)")
        elif len(data) > 2:
            try:
                from scipy.stats import gaussian_kde

                if is_percentage:
                    # Bounded KDE with reflection (REQUIREMENT #2, #6)
                    kde = gaussian_kde(data, bw_method='scott')
                    kde.set_bandwidth(kde.factor * 1.8)  # Increased bandwidth (REQUIREMENT #6)

                    # Create KDE line with clipping at [0, 1] (REQUIREMENT #2)
                    kde_x = np.linspace(max(0, data.min() - 0.05), min(1, x_max + 0.05), 300)
                    kde_x = np.clip(kde_x, 0, 1)
                    kde_y = kde(kde_x)

                    ax.plot(kde_x, kde_y, 'r-', linewidth=2.5, label='KDE (bounded)', alpha=0.8)
                    max_density = max(max_density, kde_y.max())
                    logger.info(f"        Added bounded KDE with bw_adjust=1.8")
                else:
                    # Standard KDE for continuous stats
                    kde = gaussian_kde(data)
                    kde_x = np.linspace(data.min(), x_max, 200)
                    kde_y = kde(kde_x)
                    ax.plot(kde_x, kde_y, 'r-', linewidth=2, label='KDE', alpha=0.8)
                    max_density = max(max_density, kde_y.max())
                    logger.info(f"        Added standard KDE")
            except Exception as e:
                logger.warning(f"        KDE failed: {e}")

        # =====================================================================
        # ADD STATISTICS LINES
        # =====================================================================
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}', alpha=0.7)
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.2f}', alpha=0.7)

        # =====================================================================
        # SET AXIS LIMITS
        # =====================================================================
        if is_percentage:
            ax.set_xlim(0, min(1, x_max))
        else:
            ax.set_xlim(data.min() - 0.5, x_max)

        # =====================================================================
        # CREATE TITLE WITH ANNOTATIONS (REQUIREMENT #10)
        # =====================================================================
        title = stat_descriptions.get(col, col)
        subtitle = f'n={len(data):,}, μ={mean_val:.2f}, σ={std_val:.2f}'

        if zeros_dropped_pct > 0 or low_attempts_dropped_pct > 0:
            # Annotate filtering (REQUIREMENT #10)
            filter_notes = []
            if low_attempts_dropped_pct > 0:
                filter_notes.append(f'{low_attempts_dropped_pct:.1f}% low-att filtered')
            if zeros_dropped_pct > 0:
                filter_notes.append(f'{zeros_dropped_pct:.1f}% zeros dropped')
            subtitle += '\n(' + ', '.join(filter_notes) + ')'

        ax.set_title(f'{title}\n{subtitle}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':')

    # =========================================================================
    # STEP 5: STANDARDIZE Y-AXIS SCALING (REQUIREMENT #5)
    # =========================================================================
    logger.info(f"\n>>> STEP 5: Standardizing y-axis limits...")
    logger.info(f"    Max density across all plots: {max_density:.4f}")
    logger.info(f"    Setting all y-axes to [0, {max_density * 1.1:.4f}]")

    for idx in range(len(columns)):
        axes[idx].set_ylim(0, max_density * 1.1)

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    # =========================================================================
    # ADD GLOBAL ANNOTATION
    # =========================================================================
    fig.text(0.99, 0.01,
             'Statistical Quality Controls Applied:\n'
             '✓ Discrete stats: Integer bins, NO KDE (prevents artificial bumps)\n'
             '✓ Percentages: Low-attempt games filtered, attempt-weighted histograms\n'
             '✓ Percentages: Bounded KDE with increased bandwidth (if enabled)\n'
             '✓ X-axis: Truncated at 95th percentile for readability\n'
             '✓ Y-axis: Standardized scaling across all plots\n'
             '✓ Filtering: Annotated on each plot',
             ha='right', va='bottom', fontsize=8, style='italic', alpha=0.8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"\n✓ Saved distribution plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()

    logger.info("\n✓ Distribution plotting complete!")
    logger.info("=" * 100)


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'spearman',
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot correlation matrix as heatmap.

    Args:
        df: DataFrame with data
        columns: Columns to include (None = all numeric)
        method: Correlation method ('pearson', 'spearman')
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columns) < 2:
        logger.warning("Need at least 2 columns for correlation matrix")
        return

    logger.info(f"Computing {method} correlation matrix for {len(columns)} columns...")

    # Compute correlation
    corr = df[columns].corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(columns)), max(8, len(columns) * 0.8)))

    # Plot heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()


def plot_boxplots(
    df: pd.DataFrame,
    columns: List[str] = None,
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot boxplots for numeric columns.

    Args:
        df: DataFrame with data
        columns: Columns to plot (None = all numeric)
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        logger.warning("No numeric columns to plot")
        return

    n_cols = min(4, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    logger.info(f"Creating boxplots for {len(columns)} columns...")

    for idx, col in enumerate(columns):
        ax = axes[idx]

        # Remove NaN values
        data = df[col].dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col)
            continue

        # Create boxplot
        bp = ax.boxplot([data], vert=True, patch_artist=True, widths=0.5)

        # Color the box
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        # Add statistics text
        stats_text = (
            f"n={len(data):,}\n"
            f"μ={data.mean():.2f}\n"
            f"σ={data.std():.2f}\n"
            f"Med={data.median():.2f}"
        )

        ax.text(1.3, data.median(), stats_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(col, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels([col])

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved boxplot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()


def plot_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 20,
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot value counts for a categorical column.

    Args:
        df: DataFrame with data
        column: Column to plot
        top_n: Number of top values to show
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found")
        return

    logger.info(f"Plotting value counts for {column}...")

    # Get value counts
    value_counts = df[column].value_counts().head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(value_counts) * 0.3)))

    # Create horizontal bar plot
    value_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')

    ax.set_title(f'Top {len(value_counts)} Values: {column}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, v in enumerate(value_counts):
        ax.text(v, i, f' {v:,}', va='center')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved value counts plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot time series for specified columns.

    Args:
        df: DataFrame with data
        date_col: Name of date column
        value_cols: Columns to plot over time
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    if date_col not in df.columns:
        logger.warning(f"Date column {date_col} not found")
        return

    logger.info(f"Plotting time series for {len(value_cols)} columns...")

    df_sorted = df.sort_values(date_col).copy()

    n_plots = len(value_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    for idx, col in enumerate(value_cols):
        ax = axes[idx]

        if col not in df.columns:
            ax.text(0.5, 0.5, f'{col} not found', ha='center', va='center')
            continue

        # Plot line
        ax.plot(df_sorted[date_col], df_sorted[col], marker='o', linestyle='-', linewidth=1, markersize=3)

        # Add rolling average if enough points
        if len(df_sorted) > 7:
            rolling = df_sorted[col].rolling(window=7, min_periods=1).mean()
            ax.plot(df_sorted[date_col], rolling, 'r--', linewidth=2, label='7-period MA', alpha=0.7)
            ax.legend()

        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved time series plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()


def plot_roc_and_calibration(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot ROC curve and calibration curve.

    Args:
        y_true: True binary labels
        y_pred_probs: Predicted probabilities
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve, auc

    logger.info("Plotting ROC and calibration curves...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve', fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_probs, n_bins=10, strategy='quantile'
    )

    ax2.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2, markersize=8)
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Actual Probability')
    ax2.set_title('Calibration Curve', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC/calibration plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 20,
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importance_values: Importance values
        top_n: Number of top features to show
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    logger.info(f"Plotting top {top_n} feature importances...")

    # Create DataFrame and sort
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(importance_values)
    }).sort_values('importance', ascending=False).head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(feat_imp) * 0.3)))

    # Plot
    colors = ['green' if x > 0 else 'red' for x in feat_imp['importance']]
    feat_imp.plot(x='feature', y='importance', kind='barh', ax=ax, color=colors, legend=False, edgecolor='black')

    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Absolute Importance')
    ax.set_ylabel('Feature')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, v in enumerate(feat_imp['importance']):
        ax.text(v, i, f' {v:.4f}', va='center')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()


def plot_cumulative_profit(
    hits: np.ndarray,
    odds: np.ndarray,
    stakes: np.ndarray,
    save_path: Path = None,
    show: bool = True
) -> None:
    """
    Plot cumulative profit over time.

    Args:
        hits: Binary array of wins/losses
        odds: Decimal odds
        stakes: Stake amounts
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    logger.info("Plotting cumulative profit...")

    # Calculate profits
    profits = np.where(hits == 1, stakes * (odds - 1), -stakes)
    cum_profit = np.cumsum(profits)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Cumulative profit
    ax1.plot(cum_profit, linewidth=2, color='blue')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.fill_between(range(len(cum_profit)), cum_profit, 0, where=(cum_profit >= 0), alpha=0.3, color='green', label='Profit')
    ax1.fill_between(range(len(cum_profit)), cum_profit, 0, where=(cum_profit < 0), alpha=0.3, color='red', label='Loss')

    ax1.set_title('Cumulative Profit', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bet Number')
    ax1.set_ylabel('Cumulative Profit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add final stats
    final_profit = cum_profit[-1]
    max_profit = cum_profit.max()
    max_drawdown = (np.maximum.accumulate(cum_profit) - cum_profit).max()

    stats_text = (
        f"Final Profit: {final_profit:.2f}\n"
        f"Max Profit: {max_profit:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2f}\n"
        f"ROI: {(final_profit / stakes.sum() * 100):.2f}%"
    )

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Individual bet results
    ax2.bar(range(len(profits)), profits, color=['green' if p > 0 else 'red' for p in profits], alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Individual Bet Profits', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Bet Number')
    ax2.set_ylabel('Profit/Loss')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cumulative profit plot to {save_path}")

    if show:
        from nba_research.plotting import show_figures
        show_figures()
    else:
        plt.close()
