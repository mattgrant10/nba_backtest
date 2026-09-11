"""
Visual Manager Module.

Handles pop-up plots during pipeline execution and live visual feedback.
Console-first policy: all plots must visibly appear during execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for pop-ups

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ..config import get_logger, FIGURES_DIR


logger = get_logger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plot display."""
    figsize: Tuple[int, int] = (12, 6)
    dpi: int = 100
    style: str = "whitegrid"
    show_immediately: bool = True
    save_plots: bool = True
    output_dir: Path = FIGURES_DIR
    block_on_show: bool = False  # Don't block execution


class VisualManager:
    """
    Manages visualization during pipeline execution.

    Ensures plots pop up visibly during execution and supports
    live updating visuals for long-running operations.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._figures: List[Dict] = []

        # Set style
        sns.set_style(self.config.style)
        plt.rcParams['figure.figsize'] = self.config.figsize
        plt.rcParams['figure.dpi'] = self.config.dpi

        # Enable interactive mode for live updates
        plt.ion()

    def create_figure(
        self,
        title: str,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Figure, Any]:
        """Create a new figure for plotting."""
        figsize = figsize or self.config.figsize
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        return fig, axes

    def show_and_save(
        self,
        fig: Figure,
        title: str,
        filename: Optional[str] = None
    ) -> None:
        """Show figure immediately and optionally save."""
        # Store for report export
        self._figures.append({
            "title": title,
            "figure": fig,
            "filename": filename,
        })

        # Save if configured
        if self.config.save_plots and filename:
            filepath = self.config.output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved plot: {filepath}")

        # Show plot (non-blocking)
        if self.config.show_immediately:
            fig.show()
            plt.pause(0.1)  # Allow plot to render

    def show_distribution(
        self,
        data: Union[pd.Series, np.ndarray],
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Density",
        bins: int = 50,
        show_stats: bool = True,
        clip_percentile: Optional[float] = None,
        filename: Optional[str] = None,
        use_kde: bool = True,
        data_type: Optional[str] = None,
    ) -> Figure:
        """
        Show a distribution using seaborn KDE plot with histogram underlay.

        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label (auto-generated if data_type provided)
            ylabel: Y-axis label
            bins: Number of histogram bins
            show_stats: Whether to show statistics overlay
            clip_percentile: Percentile for clipping extreme values
            filename: Output filename
            use_kde: Whether to use KDE (True) or just histogram (False)
            data_type: Type of data for auto-labeling ('probability', 'points',
                      'odds', 'count', 'percentage', 'correlation')
        """
        fig, ax = self.create_figure(title)

        # Clean data
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(data)
        data_clean = data[~np.isnan(data)]

        if len(data_clean) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            fig.tight_layout()
            self.show_and_save(fig, title, filename)
            return fig

        # Clip extreme values if requested
        if clip_percentile:
            lower = np.percentile(data_clean, clip_percentile)
            upper = np.percentile(data_clean, 100 - clip_percentile)
            data_clean = np.clip(data_clean, lower, upper)

        # Auto-generate axis labels based on data type
        xlabel, ylabel, x_format = self._get_axis_labels(data_type, xlabel, data_clean)

        # Plot using seaborn KDE with histogram
        if use_kde and len(data_clean) > 10:
            # KDE plot with filled area
            sns.kdeplot(
                data=data_clean,
                ax=ax,
                fill=True,
                alpha=0.4,
                linewidth=2,
                color='steelblue',
                label='Density Estimate'
            )
            # Add histogram underneath for reference
            ax.hist(
                data_clean,
                bins=bins,
                density=True,
                alpha=0.3,
                color='gray',
                edgecolor='darkgray',
                label='Histogram'
            )
            # Add rug plot for individual observations (if not too many)
            if len(data_clean) < 500:
                sns.rugplot(data=data_clean, ax=ax, alpha=0.3, color='steelblue')
        else:
            # Fallback to histogram for small datasets
            ax.hist(data_clean, bins=bins, density=True, edgecolor='black', alpha=0.7)

        # Set axis labels
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

        # Format x-axis based on data type
        if x_format == 'percent':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        elif x_format == 'decimal2':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        # Add gridlines for readability
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add vertical lines for key statistics
        mean_val = np.mean(data_clean)
        median_val = np.median(data_clean)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_val:.2f}', alpha=0.8)
        ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5,
                   label=f'Median: {median_val:.2f}', alpha=0.8)

        # Add stats box
        if show_stats:
            q25 = np.percentile(data_clean, 25)
            q75 = np.percentile(data_clean, 75)
            stats_text = (
                f"n = {len(data_clean):,}\n"
                f"Mean = {mean_val:.3f}\n"
                f"Std Dev = {np.std(data_clean):.3f}\n"
                f"Median = {median_val:.3f}\n"
                f"IQR = [{q25:.2f}, {q75:.2f}]\n"
                f"Range = [{data_clean.min():.2f}, {data_clean.max():.2f}]"
            )
            ax.text(0.97, 0.97, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                    fontsize=9,
                    fontfamily='monospace')

        # Add legend
        ax.legend(loc='upper left', fontsize=9)

        fig.tight_layout()
        self.show_and_save(fig, title, filename)
        return fig

    def _get_axis_labels(
        self,
        data_type: Optional[str],
        default_xlabel: str,
        data: np.ndarray
    ) -> Tuple[str, str, Optional[str]]:
        """
        Generate specific axis labels based on data type.

        Returns:
            (xlabel, ylabel, x_format)
        """
        if data_type is None:
            return default_xlabel, "Probability Density", None

        data_type = data_type.lower()

        axis_configs = {
            'probability': (
                "Hit Probability (0 = never hits, 1 = always hits)",
                "Probability Density",
                'percent'
            ),
            'points': (
                "Points per Game",
                "Probability Density",
                'decimal2'
            ),
            'assists': (
                "Assists per Game",
                "Probability Density",
                'decimal2'
            ),
            'rebounds': (
                "Rebounds per Game",
                "Probability Density",
                'decimal2'
            ),
            'fg3m': (
                "Three-Pointers Made per Game",
                "Probability Density",
                'decimal2'
            ),
            'odds': (
                "Minimum Decimal Odds for +EV (1.0 = even money)",
                "Probability Density",
                'decimal2'
            ),
            'correlation': (
                "Correlation Coefficient (-1 = inverse, 0 = none, +1 = perfect)",
                "Probability Density",
                'decimal2'
            ),
            'percentage': (
                "Percentage (%)",
                "Probability Density",
                'percent'
            ),
            'count': (
                "Count",
                "Probability Density",
                None
            ),
            'roi': (
                "Return on Investment (ROI)",
                "Probability Density",
                'percent'
            ),
            'threshold': (
                "Statistical Threshold Value",
                "Probability Density",
                'decimal2'
            ),
            'forecast': (
                "Forecasted Value (from Monte Carlo simulation)",
                "Probability Density",
                'decimal2'
            ),
            'uncertainty': (
                "Forecast Uncertainty (Standard Deviation)",
                "Probability Density",
                'decimal2'
            ),
            'rest_days': (
                "Days of Rest Between Games",
                "Probability Density",
                None
            ),
            'legs': (
                "Number of Legs in Bet Builder",
                "Probability Density",
                None
            ),
        }

        if data_type in axis_configs:
            return axis_configs[data_type]

        return default_xlabel, "Probability Density", None

    def show_grouped_bars(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        title: str = "Bar Chart",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Figure:
        """Show a grouped bar chart."""
        fig, ax = self.create_figure(title)

        if hue:
            df.pivot(index=x, columns=hue, values=y).plot(kind='bar', ax=ax)
        else:
            df.set_index(x)[y].plot(kind='bar', ax=ax)

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.legend(title=hue if hue else None)

        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        self.show_and_save(fig, title, filename)
        return fig

    def show_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Heatmap",
        annot: bool = True,
        cmap: str = "RdBu_r",
        center: float = 0,
        filename: Optional[str] = None,
    ) -> Figure:
        """Show a heatmap (useful for correlation matrices)."""
        fig, ax = self.create_figure(title, figsize=(10, 8))

        sns.heatmap(
            data,
            annot=annot,
            cmap=cmap,
            center=center,
            ax=ax,
            fmt='.2f',
            linewidths=0.5,
        )

        fig.tight_layout()
        self.show_and_save(fig, title, filename)
        return fig

    def show_line_with_bands(
        self,
        x: np.ndarray,
        y_mean: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        title: str = "Line with Confidence Bands",
        xlabel: str = "X",
        ylabel: str = "Y",
        filename: Optional[str] = None,
    ) -> Figure:
        """Show a line plot with confidence/uncertainty bands."""
        fig, ax = self.create_figure(title)

        ax.plot(x, y_mean, 'b-', linewidth=2, label='Mean')
        ax.fill_between(x, y_lower, y_upper, alpha=0.3, label='Uncertainty')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        fig.tight_layout()
        self.show_and_save(fig, title, filename)
        return fig

    def show_multi_histogram(
        self,
        data_dict: Dict[str, np.ndarray],
        title: str = "Multiple Distributions",
        xlabel: str = "Value",
        bins: int = 30,
        filename: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> Figure:
        """Show multiple KDE plots in a grid."""
        n_plots = len(data_dict)
        ncols = min(2, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = self.create_figure(title, nrows=nrows, ncols=ncols,
                                        figsize=(7 * ncols, 5 * nrows))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Get axis labels
        xlabel_full, ylabel_full, x_format = self._get_axis_labels(data_type, xlabel, np.array([]))

        for i, (name, data) in enumerate(data_dict.items()):
            if i < len(axes):
                ax = axes[i]
                data_clean = np.array(data)[~np.isnan(data)] if hasattr(data, '__len__') else np.array([data])

                if len(data_clean) > 10:
                    # KDE plot
                    sns.kdeplot(data=data_clean, ax=ax, fill=True, alpha=0.4,
                               linewidth=2, color='steelblue')
                    ax.hist(data_clean, bins=bins, density=True, alpha=0.3,
                           color='gray', edgecolor='darkgray')
                else:
                    ax.hist(data_clean, bins=bins, density=True,
                           edgecolor='black', alpha=0.7)

                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.set_xlabel(xlabel_full, fontsize=10)
                ax.set_ylabel(ylabel_full, fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_ylim(bottom=0)

                # Add mean line
                if len(data_clean) > 0:
                    mean_val = np.mean(data_clean)
                    ax.axvline(mean_val, color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.2f}')
                    ax.legend(fontsize=8)

        # Hide unused axes
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        self.show_and_save(fig, title, filename)
        return fig

    def get_all_figures(self) -> List[Dict]:
        """Return all figures created during this session."""
        return self._figures

    def clear_figures(self) -> None:
        """Clear figure history and close all plots."""
        plt.close('all')
        self._figures = []


class LiveMonteCarloVisual:
    """
    Live visual feedback for Monte Carlo simulations.

    Updates plots in real-time as simulations progress.
    """

    def __init__(
        self,
        n_simulations: int,
        update_interval: int = 100,
        figsize: Tuple[int, int] = (14, 6)
    ):
        self.n_simulations = n_simulations
        self.update_interval = update_interval
        self.figsize = figsize

        # Data storage
        self.means: List[float] = []
        self.stds: List[float] = []
        self.samples: List[float] = []
        self.iterations: List[int] = []

        # Create figure
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=figsize)
        self.fig.suptitle("Monte Carlo Simulation - Live Progress", fontsize=12)

        # Initialize plots
        self._init_plots()

    def _init_plots(self) -> None:
        """Initialize the three subplots."""
        # Plot 1: Running mean convergence
        self.ax_mean = self.axes[0]
        self.ax_mean.set_title("Mean Convergence")
        self.ax_mean.set_xlabel("Iteration")
        self.ax_mean.set_ylabel("Running Mean")
        self.line_mean, = self.ax_mean.plot([], [], 'b-', linewidth=1.5)

        # Plot 2: Running std convergence
        self.ax_std = self.axes[1]
        self.ax_std.set_title("Uncertainty Convergence")
        self.ax_std.set_xlabel("Iteration")
        self.ax_std.set_ylabel("Running Std")
        self.line_std, = self.ax_std.plot([], [], 'r-', linewidth=1.5)

        # Plot 3: Sample histogram (updates periodically)
        self.ax_hist = self.axes[2]
        self.ax_hist.set_title("Sample Distribution")
        self.ax_hist.set_xlabel("Value")
        self.ax_hist.set_ylabel("Frequency")

        self.fig.tight_layout()
        self.fig.show()

    def update(
        self,
        iteration: int,
        current_mean: float,
        current_std: float,
        new_samples: Optional[np.ndarray] = None
    ) -> None:
        """Update the live plots with new data."""
        self.iterations.append(iteration)
        self.means.append(current_mean)
        self.stds.append(current_std)

        if new_samples is not None:
            self.samples.extend(new_samples.tolist()[:100])  # Keep limited samples

        # Update only at intervals
        if iteration % self.update_interval == 0 or iteration == self.n_simulations:
            self._redraw()

    def _redraw(self) -> None:
        """Redraw all plots."""
        # Update mean line
        self.line_mean.set_data(self.iterations, self.means)
        self.ax_mean.relim()
        self.ax_mean.autoscale_view()

        # Update std line
        self.line_std.set_data(self.iterations, self.stds)
        self.ax_std.relim()
        self.ax_std.autoscale_view()

        # Update histogram
        if self.samples:
            self.ax_hist.clear()
            self.ax_hist.set_title("Sample Distribution")
            self.ax_hist.hist(self.samples[-1000:], bins=30, edgecolor='black', alpha=0.7)
            self.ax_hist.set_xlabel("Value")
            self.ax_hist.set_ylabel("Frequency")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def finalize(self, save_path: Optional[Path] = None) -> None:
        """Finalize the visualization."""
        self._redraw()

        if save_path:
            self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved MC convergence plot: {save_path}")

        plt.ioff()


# Convenience functions using default manager

_default_manager = VisualManager()


def show_distribution(
    data: Union[pd.Series, np.ndarray],
    title: str,
    xlabel: str = "Value",
    data_type: Optional[str] = None,
    **kwargs
) -> Figure:
    """
    Show a distribution using seaborn KDE plot.

    Args:
        data: Data to plot
        title: Plot title
        xlabel: X-axis label (overridden if data_type is provided)
        data_type: Type of data for auto-labeling. Options:
            - 'probability': Hit probability (0-1 scale)
            - 'points', 'assists', 'rebounds', 'fg3m': Stat types
            - 'odds': Decimal betting odds
            - 'correlation': Correlation coefficient
            - 'roi': Return on investment
            - 'forecast': Monte Carlo forecast
            - 'uncertainty': Forecast uncertainty
            - 'rest_days': Days of rest
            - 'legs': Number of bet builder legs
        **kwargs: Additional arguments passed to show_distribution
    """
    return _default_manager.show_distribution(data, title, xlabel, data_type=data_type, **kwargs)


def show_heatmap(
    data: pd.DataFrame,
    title: str = "Heatmap",
    **kwargs
) -> Figure:
    """Show a heatmap."""
    return _default_manager.show_heatmap(data, title, **kwargs)


def show_histogram(
    data: Union[pd.Series, np.ndarray],
    title: str,
    data_type: Optional[str] = None,
    **kwargs
) -> Figure:
    """Alias for show_distribution with KDE."""
    return show_distribution(data, title, data_type=data_type, **kwargs)
