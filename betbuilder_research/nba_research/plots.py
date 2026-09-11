"""Preserved research calculations: plots."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import get_logger
from .plotting import dataset_title

logger = get_logger(__name__)


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
        "Volume Stats": ["PTS", "AST", "REB", "FG3M"],
        "Shooting Stats": ["FGA", "FG3A", "FTA", "MIN"],
        "Shooting %": ["FG_PCT", "FG3_PCT", "FT_PCT"],
        "Other Stats": ["BLK", "STL", "TOV", "PLUS_MINUS"],
    }

    for category, cols in variables.items():
        # Filter to available columns
        available_cols = [c for c in cols if c in df.columns]

        if not available_cols:
            continue

        # Create batches
        for batch_num in range(0, len(available_cols), batch_size):
            batch_cols = available_cols[batch_num : batch_num + batch_size]

            n_plots = len(batch_cols)
            n_rows = 2
            n_cols = 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
            axes = axes.flatten()

            logger.info(f"\n>>> Creating {category} plots - Batch {batch_num // batch_size + 1}")

            for idx, col in enumerate(batch_cols):
                ax = axes[idx]
                data = df[col].dropna()

                if len(data) < 2:
                    continue

                # Plot histogram with KDE
                sns.histplot(
                    data,
                    kde=True,
                    ax=ax,
                    color="steelblue",
                    alpha=0.6,
                    stat="density",
                    bins=30,
                    edgecolor="black",
                    linewidth=1.2,
                )

                # Enhance KDE line
                kde_line = ax.lines[0] if ax.lines else None
                if kde_line:
                    kde_line.set_color("darkred")
                    kde_line.set_linewidth(2.5)
                    kde_line.set_alpha(0.8)

                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()

                ax.axvline(
                    mean_val,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.2f}",
                    alpha=0.8,
                )
                ax.axvline(
                    median_val,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_val:.2f}",
                    alpha=0.8,
                )

                # Add title with stats
                ax.set_title(
                    dataset_title(
                        f"{col} Distribution\nn={len(data):,} | μ={mean_val:.2f} | σ={std_val:.2f}"
                    ),
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xlabel(f"{col} Value", fontsize=10, fontweight="bold")
                ax.set_ylabel("Density", fontsize=10, fontweight="bold")
                ax.legend(fontsize=9, loc="upper right")
                ax.grid(True, alpha=0.3, linestyle=":")

                # Add annotation for key features
                q25, q75 = data.quantile([0.25, 0.75])
                skew = data.skew()

                annotation = f"IQR: [{q25:.2f}, {q75:.2f}]\n"
                if abs(skew) > 1:
                    annotation += f"Highly skewed (γ={skew:.2f})"
                elif abs(skew) > 0.5:
                    annotation += f"Moderately skewed (γ={skew:.2f})"
                else:
                    annotation += f"Approximately symmetric (γ={skew:.2f})"

                ax.text(
                    0.02,
                    0.98,
                    annotation,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

                logger.info(
                    f"    {col}: Mean={mean_val:.2f}, Median={median_val:.2f}, Skew={skew:.2f}"
                )

            # Hide unused subplots
            for idx in range(n_plots, len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(
                dataset_title(
                    f"{category} - Batch {batch_num // batch_size + 1}\n"
                    f"Kernel Density Estimation with Histogram"
                ),
                fontsize=16,
                fontweight="bold",
                y=0.995,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.99])

            # Save figure
            filename = (
                f"distributions_{category.replace(' ', '_')}_batch{batch_num // batch_size + 1}.png"
            )
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
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
    top_pairs = copula_results.nlargest(6, "Spearman_ρ", keep="all")

    # Create 2x3 grid for top 6 pairs
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    logger.info(f"\n>>> Creating scatter plots for top {len(top_pairs)} dependencies")

    for idx, (_, pair) in enumerate(top_pairs.iterrows()):
        if idx >= 6:
            break

        ax = axes[idx]
        var1, var2 = pair["Var1"], pair["Var2"]

        # Get data
        plot_data = df[[var1, var2]].dropna()

        if len(plot_data) < 10:
            continue

        # Create scatter with density
        x = plot_data[var1].values
        y = plot_data[var2].values

        # Joint plot
        ax.scatter(x, y, alpha=0.3, s=20, color="steelblue", edgecolor="none")

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_trend, p(x_trend), "r-", linewidth=2, alpha=0.8, label="Trend line")

        # Add correlation info
        pearson_r = pair["Pearson_r"]
        spearman_rho = pd.to_numeric(pair["Spearman_ρ"], errors="coerce")
        if not np.isfinite(spearman_rho):
            logger.warning(f"Invalid Spearman rho for {var1} vs {var2}, defaulting to 0.")
            spearman_rho = 0.0
        dep_type = pair["Dependency_Type"]

        # Title with correlation info
        ax.set_title(
            dataset_title(
                f"{var1} vs {var2}\nρ_s={spearman_rho:.2f} | r_p={pearson_r:.2f} | {dep_type}"
            ),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(var1, fontsize=10, fontweight="bold")
        ax.set_ylabel(var2, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

        # Add tail dependency annotation
        upper_tail = pair.get("Upper_Tail_Dep", np.nan)
        lower_tail = pair.get("Lower_Tail_Dep", np.nan)

        annotation = "Conditional correlations:\n"
        if not np.isnan(upper_tail):
            annotation += f"Upper (top 10%): {upper_tail:.2f}\n"
        if not np.isnan(lower_tail):
            annotation += f"Lower (bot 10%): {lower_tail:.2f}"

        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        )

        logger.info(f"    {var1} ↔ {var2}: ρ={spearman_rho:.2f}, type={dep_type}")

    # Hide unused subplots
    for idx in range(len(top_pairs), 6):
        axes[idx].set_visible(False)

    fig.suptitle(
        dataset_title(
            "Copula Dependency Analysis\nTop Variable Pairs by Spearman Rank Correlation"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    filename = "copula_dependencies.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    saved_figures.append(filepath)
    logger.info(f"    Saved: {filename}")

    plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} copula visualization(s)")
    return saved_figures


def create_additional_copula_plots(
    df: pd.DataFrame, copula_results: pd.DataFrame, output_dir: Path
):
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
    key_vars = ["PTS", "AST", "REB", "FG_PCT", "FG3_PCT", "MIN", "TOV", "STL", "BLK"]
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
    top_pairs = copula_results.nlargest(6, "Spearman_ρ", keep="all").head(6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (_, pair) in enumerate(top_pairs.iterrows()):
        if idx >= 6:
            break

        ax = axes[idx]
        var1, var2 = pair["Var1"], pair["Var2"]

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
        hexbin = ax.hexbin(
            x_rank, y_rank, gridsize=30, cmap="Blues", alpha=0.7, edgecolors="black", linewidths=0.2
        )

        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label("Count", fontsize=9)

        # Add correlation info
        spearman_rho = pair["Spearman_ρ"]
        dep_type = pair["Dependency_Type"]

        ax.set_title(
            dataset_title(
                f"{var1} vs {var2} (Rank-Transformed)\n"
                f"Copula Structure | ρ_s={spearman_rho:.2f} | {dep_type}"
            ),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(f"{var1} Rank (Uniform)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{var2} Rank (Uniform)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":")

        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, alpha=0.5, label="Equal-rank diagonal")
        ax.legend(fontsize=8)

        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        logger.info(f"    Rank-transformed: {var1} ↔ {var2}")

    # Hide unused subplots
    for idx in range(len(top_pairs), 6):
        axes[idx].set_visible(False)

    fig.suptitle(
        dataset_title(
            "Copula Structure Analysis (Rank-Transformed)\n"
            "Hexbin Density Plots Showing Pure Dependence Structure"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    filename = "copula_rank_transformed.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    saved_figures.append(filepath)
    logger.info(f"    Saved: {filename}")

    plt.close()

    # =========================================================================
    # 2. COPULA DEPENDENCY HEATMAP
    # =========================================================================
    logger.info("\n>>> Creating copula dependency heatmap...")

    # Create correlation matrix from copula results
    unique_vars = sorted(set(copula_results["Var1"].tolist() + copula_results["Var2"].tolist()))

    corr_matrix = pd.DataFrame(1.0, index=unique_vars, columns=unique_vars)

    for _, row in copula_results.iterrows():
        var1, var2 = row["Var1"], row["Var2"]
        rho = row["Spearman_ρ"]
        corr_matrix.loc[var1, var2] = rho
        corr_matrix.loc[var2, var1] = rho

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
        annot_kws={"fontsize": 9},
    )

    ax.set_title(
        dataset_title("Copula Dependency Heatmap\nSpearman Rank Correlation Matrix"),
        fontsize=16,
        fontweight="bold",
    )

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    filename = "copula_heatmap.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    saved_figures.append(filepath)
    logger.info(f"    Saved: {filename}")

    plt.close()

    # =========================================================================
    # 3. TAIL DEPENDENCY COMPARISON PLOT
    # =========================================================================
    logger.info("\n>>> Creating tail dependency comparison plot...")

    # Get pairs with tail dependency data
    tail_pairs = (
        copula_results[
            copula_results["Upper_Tail_Dep"].notna() & copula_results["Lower_Tail_Dep"].notna()
        ]
        .nlargest(10, "Spearman_ρ", keep="all")
        .head(10)
    )

    if len(tail_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Prepare data
        pair_labels = [f"{row['Var1']}-{row['Var2']}" for _, row in tail_pairs.iterrows()]
        upper_tail = tail_pairs["Upper_Tail_Dep"].values
        lower_tail = tail_pairs["Lower_Tail_Dep"].values

        y_pos = np.arange(len(pair_labels))

        # Plot 1: Upper vs Lower Tail Dependencies
        ax1.barh(
            y_pos - 0.2,
            upper_tail,
            0.4,
            label="Upper Tail (Top 10%)",
            color="red",
            alpha=0.7,
            edgecolor="black",
        )
        ax1.barh(
            y_pos + 0.2,
            lower_tail,
            0.4,
            label="Lower Tail (Bottom 10%)",
            color="blue",
            alpha=0.7,
            edgecolor="black",
        )

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pair_labels, fontsize=9)
        ax1.set_xlabel("Correlation Coefficient", fontsize=11, fontweight="bold")
        ax1.set_title(
            dataset_title("Tail Dependencies Comparison\nUpper vs Lower Tail Correlations"),
            fontsize=12,
            fontweight="bold",
        )
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis="x")
        ax1.axvline(0, color="black", linewidth=0.8)

        # Add value labels
        for i, (u, l) in enumerate(zip(upper_tail, lower_tail)):
            ax1.text(u + 0.02, i - 0.2, f"{u:.2f}", va="center", fontsize=8)
            ax1.text(l + 0.02, i + 0.2, f"{l:.2f}", va="center", fontsize=8)

        # Plot 2: Tail Asymmetry
        asymmetry = upper_tail - lower_tail

        colors = ["green" if a > 0 else "orange" for a in asymmetry]
        ax2.barh(y_pos, asymmetry, color=colors, alpha=0.7, edgecolor="black")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pair_labels, fontsize=9)
        ax2.set_xlabel("Tail Asymmetry (Upper - Lower)", fontsize=11, fontweight="bold")
        ax2.set_title(
            dataset_title("Conditional Correlation Difference\nPositive = Stronger Upper Tail"),
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.axvline(0, color="black", linewidth=2)

        # Add value labels
        for i, a in enumerate(asymmetry):
            x_pos = a + (0.02 if a > 0 else -0.02)
            ha = "left" if a > 0 else "right"
            ax2.text(x_pos, i, f"{a:.2f}", va="center", ha=ha, fontsize=8, fontweight="bold")

        fig.suptitle(
            dataset_title(
                "Conditional Tail Correlation Analysis\n"
                "Extreme Value Correlations (Top 10% vs Bottom 10%)"
            ),
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        filename = "copula_tail_dependencies.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        saved_figures.append(filepath)
        logger.info(f"    Saved: {filename}")

        plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} additional copula visualization(s)")
    return saved_figures


def create_fitted_copula_plots(df: pd.DataFrame, copula_results: pd.DataFrame, output_dir: Path):
    """
    Create fitted copula visualizations showing different copula families.

    Fits and visualizes:
    - Gaussian copula
    - Clayton copula (lower tail dependence)
    - Gumbel copula (upper tail dependence)
    - Frank copula (symmetric dependence)

    Args:
        df: DataFrame with game data
        copula_results: Results from copula analysis
        output_dir: Directory to save figures

    Returns:
        List of figure file paths
    """
    logger.info("\n" + "=" * 100)
    logger.info("FITTED COPULA VISUALIZATIONS")
    logger.info("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Get top 4 pairs for copula fitting
    top_pairs = copula_results.nlargest(4, "Spearman_ρ", keep="all").head(4)

    if len(top_pairs) == 0:
        logger.warning("No pairs available for copula fitting")
        return saved_figures

    logger.info(f"\n>>> Fitting copulas for top {len(top_pairs)} variable pairs")

    # Create 2x2 grid for top 4 pairs
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    from scipy.stats import rankdata, gaussian_kde

    for idx, (_, pair) in enumerate(top_pairs.iterrows()):
        if idx >= 4:
            break

        ax = axes[idx]
        var1, var2 = pair["Var1"], pair["Var2"]

        # Get data
        plot_data = df[[var1, var2]].dropna()

        if len(plot_data) < 50:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\n({var1} vs {var2})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            continue

        x = plot_data[var1].values
        y = plot_data[var2].values

        # Rank transform to [0, 1] - this is the empirical copula
        x_rank = rankdata(x) / (len(x) + 1)
        y_rank = rankdata(y) / (len(y) + 1)

        # Plot empirical copula (actual data)
        ax.scatter(
            x_rank,
            y_rank,
            alpha=0.4,
            s=15,
            c="steelblue",
            edgecolors="none",
            label="Empirical Copula",
            zorder=2,
        )

        # Fit Gaussian copula (using Spearman correlation)
        spearman_rho = pair["Spearman_ρ"]

        # Create a grid for contour plotting
        grid_size = 50
        u_grid = np.linspace(0.01, 0.99, grid_size)
        v_grid = np.linspace(0.01, 0.99, grid_size)
        U, V = np.meshgrid(u_grid, v_grid)

        # For Gaussian copula, transform to normal quantiles
        from scipy.stats import norm

        # Gaussian copula density contours (approximate)
        # Convert uniform margins to normal quantiles
        U_norm = norm.ppf(U)
        V_norm = norm.ppf(V)

        # Bivariate normal density with correlation = spearman_rho (approximation)
        rho = float(np.clip(spearman_rho, -0.999, 0.999))
        Z = np.exp(-0.5 * (U_norm**2 + V_norm**2 - 2 * rho * U_norm * V_norm) / (1 - rho**2))
        Z = Z / (2 * np.pi * np.sqrt(1 - rho**2))

        # Plot Gaussian copula contours
        contour = ax.contour(U, V, Z, levels=8, colors="red", alpha=0.6, linewidths=1.5, zorder=3)
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.2e")

        # Add independence reference line
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.4, label="Equal-rank diagonal", zorder=1
        )

        # Get correlation info
        pearson_r = pair["Pearson_r"]
        dep_type = pair["Dependency_Type"]

        # Title with correlation info
        ax.set_title(
            dataset_title(
                f"{var1} vs {var2}\n"
                f"Illustrative surface | ρ_s={spearman_rho:.3f} | r_p={pearson_r:.3f}\n"
                f"{dep_type}"
            ),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(f"{var1} (Rank)", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{var2} (Rank)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.legend(fontsize=8, loc="upper left")

        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add tail dependency annotation
        upper_tail = pair.get("Upper_Tail_Dep", np.nan)
        lower_tail = pair.get("Lower_Tail_Dep", np.nan)

        annotation = "Conditional correlations:\n"
        if not np.isnan(upper_tail):
            annotation += f"Upper: {upper_tail:.2f}\n"
        if not np.isnan(lower_tail):
            annotation += f"Lower: {lower_tail:.2f}"

        ax.text(
            0.98,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        logger.info(f"    Fitted Gaussian copula: {var1} ↔ {var2} (ρ={spearman_rho:.3f})")

    # Hide unused subplots
    for idx in range(len(top_pairs), 4):
        axes[idx].set_visible(False)

    fig.suptitle(
        dataset_title(
            "Illustrative Surface Analysis\nApproximate surfaces (not fitted copula densities)"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    filename = "copula_fitted_gaussian.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    saved_figures.append(filepath)
    logger.info(f"\n    Saved: {filename}")

    plt.close()

    # =========================================================================
    # COPULA FAMILY COMPARISON (for top pair only)
    # =========================================================================
    logger.info("\n>>> Creating copula family comparison plot...")

    if len(top_pairs) > 0:
        top_pair = top_pairs.iloc[0]
        var1, var2 = top_pair["Var1"], top_pair["Var2"]

        plot_data = df[[var1, var2]].dropna()

        if len(plot_data) >= 50:
            x = plot_data[var1].values
            y = plot_data[var2].values

            # Rank transform
            x_rank = rankdata(x) / (len(x) + 1)
            y_rank = rankdata(y) / (len(y) + 1)

            # Create 2x2 grid for different copula families
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))

            spearman_rho = pd.to_numeric(top_pair["Spearman_ρ"], errors="coerce")
            if not np.isfinite(spearman_rho):
                logger.warning(f"Invalid Spearman rho for {var1} vs {var2}, defaulting to 0.")
                spearman_rho = 0.0

            # Subplot 1: Empirical Copula with KDE
            ax1 = axes[0, 0]
            ax1.scatter(x_rank, y_rank, alpha=0.3, s=10, c="steelblue", edgecolors="none")
            ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.4)

            grid_size = 50
            u_grid = np.linspace(0.01, 0.99, grid_size)
            v_grid = np.linspace(0.01, 0.99, grid_size)
            U, V = np.meshgrid(u_grid, v_grid)

            # Add KDE contours for empirical copula
            try:
                kde = gaussian_kde(np.vstack([x_rank, y_rank]))
                positions = np.vstack([U.ravel(), V.ravel()])
                Z_kde = np.reshape(kde(positions).T, U.shape)
                ax1.contour(U, V, Z_kde, levels=8, colors="darkblue", alpha=0.6, linewidths=1.5)
            except:
                pass

            ax1.set_title(
                dataset_title(f"Empirical Copula (KDE)\n{var1} vs {var2}"),
                fontsize=12,
                fontweight="bold",
            )
            ax1.set_xlabel(f"{var1} (Rank)", fontsize=10)
            ax1.set_ylabel(f"{var2} (Rank)", fontsize=10)
            ax1.grid(True, alpha=0.2)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

            # Subplot 2: Gaussian Copula (already computed above)
            ax2 = axes[0, 1]
            ax2.scatter(x_rank, y_rank, alpha=0.3, s=10, c="steelblue", edgecolors="none")
            ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.4)

            # Gaussian copula contours (same as before)
            rho = float(np.clip(spearman_rho, -0.999, 0.999))
            U_norm = norm.ppf(U)
            V_norm = norm.ppf(V)
            Z_gaussian = np.exp(
                -0.5 * (U_norm**2 + V_norm**2 - 2 * rho * U_norm * V_norm) / (1 - rho**2)
            )
            Z_gaussian = Z_gaussian / (2 * np.pi * np.sqrt(1 - rho**2))

            ax2.contour(U, V, Z_gaussian, levels=8, colors="red", alpha=0.6, linewidths=1.5)
            ax2.set_title(
                dataset_title(f"Gaussian Copula\nρ={spearman_rho:.3f}"),
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_xlabel(f"{var1} (Rank)", fontsize=10)
            ax2.set_ylabel(f"{var2} (Rank)", fontsize=10)
            ax2.grid(True, alpha=0.2)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

            # Subplot 3: Clayton Copula (Lower tail dependence)
            ax3 = axes[1, 0]
            ax3.scatter(x_rank, y_rank, alpha=0.3, s=10, c="steelblue", edgecolors="none")
            ax3.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.4)

            # Clayton copula parameter (approximate from Spearman)
            if spearman_rho > 0:
                theta_clayton = 2 * spearman_rho / (1 - spearman_rho)
                theta_clayton = max(0.1, theta_clayton)  # Must be positive

                # Clayton copula density (simplified visualization)
                # Higher density in lower-left corner (lower tail dependence)
                Z_clayton = (U ** (-theta_clayton) + V ** (-theta_clayton) - 1) ** (
                    -1 / theta_clayton - 1
                )
                Z_clayton = np.nan_to_num(Z_clayton, nan=0, posinf=0, neginf=0).astype(
                    float, copy=False
                )

                ax3.contour(U, V, Z_clayton, levels=8, colors="green", alpha=0.6, linewidths=1.5)
                ax3.set_title(
                    dataset_title(f"Clayton Copula\nθ={theta_clayton:.2f} | Lower Tail Dep."),
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax3.set_title(
                    dataset_title("Clayton Copula\n(Not suitable for negative correlation)"),
                    fontsize=12,
                    fontweight="bold",
                )

            ax3.set_xlabel(f"{var1} (Rank)", fontsize=10)
            ax3.set_ylabel(f"{var2} (Rank)", fontsize=10)
            ax3.grid(True, alpha=0.2)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

            # Subplot 4: Gumbel Copula (Upper tail dependence)
            ax4 = axes[1, 1]
            ax4.scatter(x_rank, y_rank, alpha=0.3, s=10, c="steelblue", edgecolors="none")
            ax4.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.4)

            # Gumbel copula parameter (approximate from Spearman)
            if spearman_rho > 0:
                theta_gumbel = 1 / (1 - spearman_rho)
                theta_gumbel = max(1.0, theta_gumbel)  # Must be >= 1

                # Gumbel copula density (simplified visualization)
                # Higher density in upper-right corner (upper tail dependence)
                Z_gumbel = np.exp(
                    -(
                        ((-np.log(U)) ** theta_gumbel + (-np.log(V)) ** theta_gumbel)
                        ** (1 / theta_gumbel)
                    )
                )
                Z_gumbel = np.nan_to_num(Z_gumbel, nan=0, posinf=0, neginf=0).astype(
                    float, copy=False
                )

                ax4.contour(U, V, Z_gumbel, levels=8, colors="purple", alpha=0.6, linewidths=1.5)
                ax4.set_title(
                    dataset_title(f"Gumbel Copula\nθ={theta_gumbel:.2f} | Upper Tail Dep."),
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax4.set_title(
                    dataset_title("Gumbel Copula\n(Not suitable for negative correlation)"),
                    fontsize=12,
                    fontweight="bold",
                )

            ax4.set_xlabel(f"{var1} (Rank)", fontsize=10)
            ax4.set_ylabel(f"{var2} (Rank)", fontsize=10)
            ax4.grid(True, alpha=0.2)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

            fig.suptitle(
                dataset_title(
                    f"Illustrative Family Comparison\n{var1} vs {var2} | Top Variable Pair"
                ),
                fontsize=16,
                fontweight="bold",
                y=0.995,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.99])

            # Save
            filename = "copula_family_comparison.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            saved_figures.append(filepath)
            logger.info(f"    Saved: {filename}")

            plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} fitted copula visualization(s)")
    return saved_figures


def create_team_consistency_plots(
    df: pd.DataFrame,
    prop_analysis: pd.DataFrame,
    output_dir: Path,
    teams_per_page: int = 1,
    players_per_plot: int = 8,
):
    """
    Create team-by-team consistency plots - ONE TEAM PER PAGE with all player names labeled.
    Splits into multiple plots if a team has more players than players_per_plot.

    Args:
        df: Game-level data
        prop_analysis: Player consistency metrics
        output_dir: Directory to save figures
        teams_per_page: Number of teams per figure (default 1 for clarity)
        players_per_plot: Maximum players per individual plot before splitting

    Returns:
        List of figure file paths
    """
    logger.info("\n" + "=" * 100)
    logger.info("TEAM CONSISTENCY VISUALIZATIONS")
    logger.info("=" * 100)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Get all teams
    teams = sorted(df["player_team"].dropna().astype(str).unique())
    total_teams = len(teams)

    logger.info(
        f"\n>>> Creating individual figures for {total_teams} teams (1 team per figure, max {players_per_plot} players per plot)"
    )

    for team_idx, team in enumerate(teams):
        # Get team players
        team_players = df[df["player_team"] == team]["player_name"].unique()

        # Get consistency data - get ALL players, not just top 15
        team_consistency = prop_analysis[prop_analysis["player_name"].isin(team_players)]
        team_consistency = team_consistency.sort_values("PPG", ascending=False)

        if len(team_consistency) == 0:
            logger.info(f"    {team}: No data available")
            continue

        # Split into multiple plots if too many players
        total_players = len(team_consistency)
        n_plots = (total_players + players_per_plot - 1) // players_per_plot

        logger.info(
            f"\n>>> Team {team_idx + 1}/{total_teams}: {team} - {total_players} players in {n_plots} plot(s)"
        )

        for plot_idx in range(n_plots):
            start_player = plot_idx * players_per_plot
            end_player = min((plot_idx + 1) * players_per_plot, total_players)
            plot_data = team_consistency.iloc[start_player:end_player]

            # Create figure for this batch
            fig, ax = plt.subplots(figsize=(16, 10))

            # Create scatter plot: Consistency vs Performance
            x = plot_data["PPG"].values
            y = plot_data["PTS_CV"].values
            names = plot_data["player_name"].values

            # Color by consistency (green = consistent, red = inconsistent)
            colors = ["green" if cv < 0.40 else "orange" if cv < 0.60 else "red" for cv in y]

            # Plot with larger markers
            ax.scatter(x, y, s=200, c=colors, alpha=0.7, edgecolor="black", linewidth=2)

            # Add ALL player labels with smart positioning to avoid overlap
            texts = []
            for i, (xi, yi, name) in enumerate(zip(x, y, names)):
                # Use full name or last name based on length
                display_name = name if len(name) < 20 else name.split()[-1]
                texts.append(
                    ax.annotate(
                        display_name,
                        (xi, yi),
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                    )
                )

            # Try to use adjustText if available for better label placement
            try:
                from adjustText import adjust_text

                adjust_text(
                    texts,
                    x=x,
                    y=y,
                    ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
                    expand_points=(1.5, 1.5),
                    force_text=(0.5, 0.5),
                )
            except ImportError:
                # Fallback: simple offset positioning if adjustText not installed
                for i, txt in enumerate(texts):
                    txt.set_position((x[i], y[i]))
                    txt.xyann = (5, 5)
                    txt.set_anncoords("offset points")

            # Add consistency threshold lines
            ax.axhline(
                0.40,
                color="blue",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Good consistency (CV=0.40)",
            )
            ax.axhline(
                0.60,
                color="orange",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label="Moderate consistency (CV=0.60)",
            )

            # Title with plot number if multiple plots
            if n_plots > 1:
                title = f"{team} - Player Consistency vs Performance\n(Plot {plot_idx + 1}/{n_plots}: Players {start_player + 1}-{end_player} by PPG)"
            else:
                title = f"{team} - Player Consistency vs Performance\n({len(plot_data)} Players)"

            ax.set_title(dataset_title(title), fontweight="bold", fontsize=14)
            ax.set_xlabel("Points Per Game (PPG)", fontsize=12, fontweight="bold")
            ax.set_ylabel(
                "Coefficient of Variation (CV) - Lower = More Consistent",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(True, alpha=0.3, linestyle=":")

            # Add color legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor="green", edgecolor="black", alpha=0.7, label="Consistent (CV < 0.40)"
                ),
                Patch(
                    facecolor="orange",
                    edgecolor="black",
                    alpha=0.7,
                    label="Moderate (0.40 ≤ CV < 0.60)",
                ),
                Patch(
                    facecolor="red", edgecolor="black", alpha=0.7, label="Inconsistent (CV ≥ 0.60)"
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

            # Annotate regions
            ax.text(
                0.98,
                0.98,
                "Inconsistent\nHigh-scorers",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="salmon", alpha=0.3),
            )
            ax.text(
                0.98,
                0.02,
                "Consistent\nHigh-scorers",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
            )

            plt.tight_layout()

            # Save with team name in filename
            team_safe = team.replace(" ", "_").replace("/", "_")
            if n_plots > 1:
                filename = f"team_consistency_{team_safe}_part{plot_idx + 1}.png"
            else:
                filename = f"team_consistency_{team_safe}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            saved_figures.append(filepath)
            logger.info(f"    Saved: {filename} ({len(plot_data)} players)")

            plt.close()

    logger.info(f"\n✓ Created {len(saved_figures)} team consistency figures")
    return saved_figures
