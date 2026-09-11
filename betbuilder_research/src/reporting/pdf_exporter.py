"""
PDF Report Exporter Module.

Generates comprehensive PDF reports with all tables, figures,
logging output, and data documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io
import textwrap

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate

from ..config import get_logger, FIGURES_DIR


logger = get_logger(__name__)


# ============================================================================
# DATA FEED DOCUMENTATION
# ============================================================================

DATA_FEED_DOCUMENTATION = """
================================================================================
DATA FEED DOCUMENTATION
================================================================================

This section documents all data sources that feed into the edge recommendations.

1. PLAYER GAME LOGS (Primary Source)
   ---------------------------------
   Source: PlayerStatistics_2018_2025.csv (or similar files in data/raw/)

   Contains historical game-by-game statistics for NBA players including:
   - Points scored (pts)
   - Three-pointers made (fg3m)
   - Assists (ast)
   - Rebounds (reb)
   - Plus other box score statistics

   This data is used to:
   - Fit statistical distributions for each player-stat combination
   - Calculate player averages and variance
   - Determine correlation structure between stats

2. STATISTICAL DISTRIBUTIONS (Derived)
   ------------------------------------
   Generated from player game logs using distribution fitting.

   For each player and stat, we fit:
   - Negative Binomial distribution (for overdispersed counts)
   - Poisson distribution (for regular count data)
   - Degenerate distribution (for near-zero averages)

   The fitted distribution captures:
   - Expected value (mean performance)
   - Variance (consistency/volatility)
   - Shape parameters for probability calculations

3. CORRELATION STRUCTURE (Derived)
   --------------------------------
   Computed from player game logs using Spearman correlation.

   Captures relationships between stats (e.g., high-scoring games
   often correlate with more assists for playmakers).

4. PROP LINES / THRESHOLDS
   ------------------------
   Either loaded from external data or synthetically generated.

   Synthetic generation uses:
   - Player rolling averages
   - Standard betting line increments (0.5, 1.5, 2.5, etc.)

   These represent the "threshold" values for Over/Under bets.

================================================================================
TERMINOLOGY GLOSSARY
================================================================================

THRESHOLD (formerly "line"):
   The statistical cutoff value for a prop bet.
   - For OVER bets: Player must EXCEED this value to hit
   - For UNDER bets: Player must stay BELOW this value to hit

   Example: "Points Threshold 24.5" means:
   - OVER 24.5 pts: Player needs 25+ points to win
   - UNDER 24.5 pts: Player needs 24 or fewer points to win

PROBABILITY:
   The model-estimated likelihood of hitting the threshold.
   Calculated from the fitted statistical distribution.

   Example: 0.65 probability = 65% chance of hitting

MIN ODDS FOR +EV (Minimum Odds for Positive Expected Value):
   The lowest decimal odds at which a bet becomes profitable.

   Formula: Min Odds = 1 / Probability

   Example: If probability = 0.50, Min Odds = 1/0.50 = 2.00

   If a sportsbook offers odds >= Min Odds, the bet has positive
   expected value (edge).

DECIMAL ODDS:
   European odds format where:
   - 1.50 = 50% implied probability (bet $100, win $150 total)
   - 2.00 = 50% implied probability (bet $100, win $200 total)
   - 3.00 = 33% implied probability (bet $100, win $300 total)

TIERS:
   Segmentation of recommendations by threshold value:
   - LOWER TIER: Smaller thresholds (easier to hit for overs)
   - MID TIER: Medium thresholds (balanced difficulty)
   - HIGHER TIER: Larger thresholds (harder to hit for overs)

================================================================================
"""


@dataclass
class PDFSection:
    """A section in the PDF report."""
    title: str
    content_type: str  # "text", "table", "figure", "log"
    content: Any
    description: Optional[str] = None


@dataclass
class PDFReportConfig:
    """Configuration for PDF report generation."""
    page_width: float = 11.0  # inches (landscape)
    page_height: float = 8.5  # inches
    margin: float = 0.75
    font_size: int = 8
    title_font_size: int = 14
    header_font_size: int = 12
    table_font_size: int = 7
    rows_per_page: int = 40
    include_logs: bool = True
    include_data_docs: bool = True


class PDFReportExporter:
    """
    Generates comprehensive PDF reports with all pipeline output.

    Includes:
    - Full tables (40 rows each)
    - All figures
    - Console/logging output
    - Data feed documentation
    - Terminology glossary
    """

    def __init__(self, config: Optional[PDFReportConfig] = None):
        self.config = config or PDFReportConfig()
        self.sections: List[PDFSection] = []
        self.console_log: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.figures: List[Tuple[plt.Figure, str, str]] = []  # (fig, title, step)
        self.tables: List[Tuple[pd.DataFrame, str, str, str]] = []  # (df, title, step, desc)

    def set_metadata(self, **kwargs) -> None:
        """Set report metadata."""
        self.metadata.update(kwargs)

    def add_log_line(self, line: str) -> None:
        """Add a line to the console log capture."""
        self.console_log.append(line)

    def add_figure(
        self,
        fig: plt.Figure,
        title: str,
        step: str = "misc"
    ) -> None:
        """Add a figure to the report."""
        self.figures.append((fig, title, step))

    def add_table(
        self,
        df: pd.DataFrame,
        title: str,
        step: str = "misc",
        description: str = ""
    ) -> None:
        """Add a table to the report."""
        self.tables.append((df.copy(), title, step, description))

    def add_text_section(self, title: str, text: str) -> None:
        """Add a text section."""
        self.sections.append(PDFSection(
            title=title,
            content_type="text",
            content=text
        ))

    def generate_pdf(self, output_path: Path) -> Path:
        """Generate the complete PDF report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(output_path) as pdf:
            # Title page
            self._add_title_page(pdf)

            # Data documentation
            if self.config.include_data_docs:
                self._add_documentation_pages(pdf)

            # Tables (grouped by step)
            self._add_table_pages(pdf)

            # Figures
            self._add_figure_pages(pdf)

            # Console log
            if self.config.include_logs and self.console_log:
                self._add_log_pages(pdf)

        logger.info(f"PDF report saved to: {output_path}")
        return output_path

    def _add_title_page(self, pdf: PdfPages) -> None:
        """Add title page with metadata."""
        fig, ax = plt.subplots(figsize=(self.config.page_width, self.config.page_height))
        ax.axis('off')

        # Title
        ax.text(0.5, 0.85, "NBA BET BUILDER PIPELINE",
                fontsize=24, fontweight='bold', ha='center', va='top',
                transform=ax.transAxes)
        ax.text(0.5, 0.78, "Comprehensive Research Report",
                fontsize=16, ha='center', va='top',
                transform=ax.transAxes)

        # Metadata
        y_pos = 0.65
        for key, value in self.metadata.items():
            display_key = key.replace('_', ' ').title()
            ax.text(0.3, y_pos, f"{display_key}:", fontsize=12, fontweight='bold',
                    ha='right', va='top', transform=ax.transAxes)
            ax.text(0.32, y_pos, str(value), fontsize=12,
                    ha='left', va='top', transform=ax.transAxes)
            y_pos -= 0.05

        # Footer
        ax.text(0.5, 0.1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                fontsize=10, ha='center', va='bottom',
                transform=ax.transAxes, style='italic')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_documentation_pages(self, pdf: PdfPages) -> None:
        """Add data feed documentation pages."""
        # Split documentation into pages
        lines = DATA_FEED_DOCUMENTATION.strip().split('\n')
        lines_per_page = 50

        for i in range(0, len(lines), lines_per_page):
            page_lines = lines[i:i + lines_per_page]

            fig, ax = plt.subplots(figsize=(self.config.page_width, self.config.page_height))
            ax.axis('off')

            text = '\n'.join(page_lines)
            ax.text(0.02, 0.98, text,
                    fontsize=self.config.font_size,
                    fontfamily='monospace',
                    ha='left', va='top',
                    transform=ax.transAxes)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _add_table_pages(self, pdf: PdfPages) -> None:
        """Add table pages."""
        for df, title, step, description in self.tables:
            self._render_table_to_pdf(pdf, df, title, step, description)

    def _render_table_to_pdf(
        self,
        pdf: PdfPages,
        df: pd.DataFrame,
        title: str,
        step: str,
        description: str
    ) -> None:
        """Render a single table (potentially multi-page) to PDF."""
        if df.empty:
            return

        rows_per_page = self.config.rows_per_page
        n_pages = (len(df) + rows_per_page - 1) // rows_per_page

        for page_num in range(n_pages):
            start_idx = page_num * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(df))
            page_df = df.iloc[start_idx:end_idx]

            fig, ax = plt.subplots(figsize=(self.config.page_width, self.config.page_height))
            ax.axis('off')

            # Header
            page_title = f"{title}"
            if n_pages > 1:
                page_title += f" (Page {page_num + 1}/{n_pages})"

            ax.text(0.5, 0.98, page_title,
                    fontsize=self.config.header_font_size,
                    fontweight='bold',
                    ha='center', va='top',
                    transform=ax.transAxes)

            ax.text(0.5, 0.94, f"Step: {step.upper()} | Rows {start_idx + 1}-{end_idx} of {len(df)}",
                    fontsize=self.config.font_size,
                    ha='center', va='top',
                    transform=ax.transAxes,
                    style='italic')

            if description and page_num == 0:
                ax.text(0.5, 0.90, description,
                        fontsize=self.config.font_size,
                        ha='center', va='top',
                        transform=ax.transAxes)

            # Table using tabulate
            table_str = tabulate(
                page_df,
                headers='keys',
                tablefmt='simple',
                showindex=False,
                floatfmt='.4f'
            )

            ax.text(0.02, 0.85, table_str,
                    fontsize=self.config.table_font_size,
                    fontfamily='monospace',
                    ha='left', va='top',
                    transform=ax.transAxes)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _add_figure_pages(self, pdf: PdfPages) -> None:
        """Add figure pages."""
        for fig, title, step in self.figures:
            # Create a new figure for the PDF page
            page_fig, ax = plt.subplots(figsize=(self.config.page_width, self.config.page_height))
            ax.axis('off')

            # Title
            ax.text(0.5, 0.98, title,
                    fontsize=self.config.header_font_size,
                    fontweight='bold',
                    ha='center', va='top',
                    transform=ax.transAxes)
            ax.text(0.5, 0.94, f"Step: {step.upper()}",
                    fontsize=self.config.font_size,
                    ha='center', va='top',
                    transform=ax.transAxes,
                    style='italic')

            # Save the original figure to a buffer and embed
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)

            from PIL import Image
            img = Image.open(buf)

            # Create an inset axes for the image
            img_ax = page_fig.add_axes([0.1, 0.1, 0.8, 0.75])
            img_ax.imshow(img)
            img_ax.axis('off')

            pdf.savefig(page_fig, bbox_inches='tight')
            plt.close(page_fig)
            buf.close()

    def _add_log_pages(self, pdf: PdfPages) -> None:
        """Add console log pages."""
        lines_per_page = 60
        all_lines = self.console_log

        for i in range(0, len(all_lines), lines_per_page):
            page_lines = all_lines[i:i + lines_per_page]

            fig, ax = plt.subplots(figsize=(self.config.page_width, self.config.page_height))
            ax.axis('off')

            # Header
            page_num = i // lines_per_page + 1
            total_pages = (len(all_lines) + lines_per_page - 1) // lines_per_page

            ax.text(0.5, 0.98, f"Console Output (Page {page_num}/{total_pages})",
                    fontsize=self.config.header_font_size,
                    fontweight='bold',
                    ha='center', va='top',
                    transform=ax.transAxes)

            # Log content
            text = '\n'.join(page_lines)
            ax.text(0.02, 0.92, text,
                    fontsize=6,
                    fontfamily='monospace',
                    ha='left', va='top',
                    transform=ax.transAxes)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


class ConsoleCapture:
    """Context manager to capture console output."""

    def __init__(self, pdf_exporter: PDFReportExporter):
        self.pdf_exporter = pdf_exporter
        self.original_stdout = None
        self.capture_buffer = None

    def __enter__(self):
        import sys
        self.original_stdout = sys.stdout
        self.capture_buffer = io.StringIO()

        # Create a tee to both capture and display
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = Tee(self.original_stdout, self.capture_buffer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout = self.original_stdout

        # Add captured content to PDF exporter
        captured = self.capture_buffer.getvalue()
        for line in captured.split('\n'):
            self.pdf_exporter.add_log_line(line)

        return False


def create_tiered_tables(
    df: pd.DataFrame,
    stat: str,
    threshold_col: str = "threshold",
    direction: str = "over"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into lower, mid, and higher tier tables.

    Tiers are based on the threshold values:
    - Lower: Bottom 33% of thresholds
    - Mid: Middle 33% of thresholds
    - Higher: Top 33% of thresholds

    Returns:
        (lower_tier_df, mid_tier_df, higher_tier_df)
    """
    if df.empty or threshold_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Calculate tier boundaries
    thresholds = df[threshold_col].dropna()
    if len(thresholds) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    q33 = thresholds.quantile(0.33)
    q66 = thresholds.quantile(0.66)

    lower_tier = df[df[threshold_col] <= q33].copy()
    mid_tier = df[(df[threshold_col] > q33) & (df[threshold_col] <= q66)].copy()
    higher_tier = df[df[threshold_col] > q66].copy()

    return lower_tier, mid_tier, higher_tier


def get_tier_description(stat: str, tier: str, direction: str) -> str:
    """Get human-readable description for a tier."""
    stat_names = {
        "pts": "Points",
        "fg3m": "Three-Pointers Made",
        "ast": "Assists",
        "reb": "Rebounds"
    }
    stat_name = stat_names.get(stat, stat.upper())

    tier_descs = {
        "lower": "Lower thresholds (easier for OVER bets to hit)",
        "mid": "Mid-range thresholds (balanced difficulty)",
        "higher": "Higher thresholds (harder for OVER bets to hit)"
    }
    tier_desc = tier_descs.get(tier, tier)

    return f"{stat_name} - {direction.upper()} - {tier.upper()} TIER\n{tier_desc}"
