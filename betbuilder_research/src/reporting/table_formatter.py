"""
Table Formatter Module.

Enforces the 5-column maximum rule and HEAD/TAIL/FOOTER display pattern.
All tables displayed in console must pass through this formatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import pandas as pd
import numpy as np
from tabulate import tabulate

from ..config import get_logger


logger = get_logger(__name__)


class Verbosity(Enum):
    """Reporting verbosity levels."""
    LITE = "lite"
    FULL = "full"


@dataclass
class TableConfig:
    """Configuration for table display."""
    max_columns: int = 5
    head_rows: int = 40
    tail_rows: int = 40
    float_precision: int = 4
    max_col_width: int = 25
    show_index: bool = False
    tablefmt: str = "simple"  # tabulate format: simple, grid, fancy_grid, pipe, etc.


@dataclass
class TableFooter:
    """Footer statistics for a table."""
    row_count: int
    col_count: int
    missing_pct: float
    duplicate_pct: float
    zero_variance_cols: List[str]
    warnings: List[str] = field(default_factory=list)


class TableFormatter:
    """
    Formats DataFrames for console display with strict constraints.

    Enforces:
    - Maximum 5 columns per table (splits if needed)
    - HEAD + TAIL + FOOTER display pattern
    - Consistent formatting and width control
    """

    def __init__(self, config: Optional[TableConfig] = None):
        self.config = config or TableConfig()
        self._printed_tables: List[Dict] = []

    def format_table(
        self,
        df: pd.DataFrame,
        title: str,
        key_columns: Optional[List[str]] = None,
        show_footer: bool = True,
    ) -> str:
        """
        Format a DataFrame for console display.

        If DataFrame has more than 5 columns, splits into multiple tables.

        Args:
            df: DataFrame to format
            title: Table title
            key_columns: Columns to check for duplicates
            show_footer: Whether to show footer statistics

        Returns:
            Formatted string ready for console output
        """
        if df.empty:
            return f"\n{title}\n{'='*len(title)}\n(Empty table)\n"

        # Split if too many columns
        if len(df.columns) > self.config.max_columns:
            return self._format_split_tables(df, title, key_columns, show_footer)

        return self._format_single_table(df, title, key_columns, show_footer)

    def _format_single_table(
        self,
        df: pd.DataFrame,
        title: str,
        key_columns: Optional[List[str]] = None,
        show_footer: bool = True,
    ) -> str:
        """Format a single table (≤5 columns)."""
        lines = []

        # Title
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

        # Prepare display DataFrame
        display_df = self._prepare_display_df(df)

        n_rows = len(display_df)
        head_n = min(self.config.head_rows, n_rows)
        tail_n = min(self.config.tail_rows, n_rows)

        # HEAD section
        if n_rows > 0:
            lines.append(f"\n[HEAD - Top {head_n} rows]")
            head_df = display_df.head(head_n)
            lines.append(tabulate(
                head_df,
                headers='keys',
                tablefmt=self.config.tablefmt,
                showindex=self.config.show_index,
                floatfmt=f".{self.config.float_precision}f"
            ))

        # TAIL section (if different from head)
        if n_rows > head_n + tail_n:
            lines.append(f"\n... ({n_rows - head_n - tail_n} rows omitted) ...")
            lines.append(f"\n[TAIL - Bottom {tail_n} rows]")
            tail_df = display_df.tail(tail_n)
            lines.append(tabulate(
                tail_df,
                headers='keys',
                tablefmt=self.config.tablefmt,
                showindex=self.config.show_index,
                floatfmt=f".{self.config.float_precision}f"
            ))
        elif n_rows > head_n:
            # Show remaining rows as tail
            remaining = n_rows - head_n
            lines.append(f"\n[TAIL - Bottom {remaining} rows]")
            tail_df = display_df.tail(remaining)
            lines.append(tabulate(
                tail_df,
                headers='keys',
                tablefmt=self.config.tablefmt,
                showindex=self.config.show_index,
                floatfmt=f".{self.config.float_precision}f"
            ))

        # FOOTER section
        if show_footer:
            footer = self._compute_footer(df, key_columns)
            lines.append(self._format_footer(footer))

        lines.append("")

        # Store for report export
        self._printed_tables.append({
            "title": title,
            "df": df.copy(),
            "footer": footer if show_footer else None,
        })

        return "\n".join(lines)

    def _format_split_tables(
        self,
        df: pd.DataFrame,
        title: str,
        key_columns: Optional[List[str]] = None,
        show_footer: bool = True,
    ) -> str:
        """Split DataFrame into multiple ≤5-column tables."""
        lines = []
        cols = list(df.columns)
        n_splits = (len(cols) + self.config.max_columns - 1) // self.config.max_columns

        for i in range(n_splits):
            start_idx = i * self.config.max_columns
            end_idx = min((i + 1) * self.config.max_columns, len(cols))
            split_cols = cols[start_idx:end_idx]
            split_df = df[split_cols]

            split_title = f"{title} ({i+1}/{n_splits})"

            # Only show footer on last split
            show_split_footer = show_footer and (i == n_splits - 1)

            lines.append(self._format_single_table(
                split_df, split_title, key_columns, show_split_footer
            ))

        return "\n".join(lines)

    def _prepare_display_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for display (truncate strings, format numbers)."""
        display_df = df.copy()

        for col in display_df.columns:
            # Truncate long strings
            if display_df[col].dtype == object:
                display_df[col] = display_df[col].astype(str).str[:self.config.max_col_width]
            # Format percentages
            elif col.endswith('_pct') or col.endswith('_rate') or 'probability' in col.lower():
                if display_df[col].dtype in [np.float64, np.float32]:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                    )

        return display_df

    def _compute_footer(
        self,
        df: pd.DataFrame,
        key_columns: Optional[List[str]] = None
    ) -> TableFooter:
        """Compute footer statistics for a table."""
        # Missingness
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0

        # Duplicates
        duplicate_pct = 0.0
        if key_columns:
            valid_keys = [k for k in key_columns if k in df.columns]
            if valid_keys:
                n_dupes = df.duplicated(subset=valid_keys).sum()
                duplicate_pct = n_dupes / len(df) * 100 if len(df) > 0 else 0

        # Zero-variance columns
        zero_var_cols = []
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if df[col].nunique() <= 1:
                    zero_var_cols.append(col)
            elif df[col].nunique() <= 1:
                zero_var_cols.append(col)

        # Warnings
        warnings = []
        if missing_pct > 10:
            warnings.append(f"High missingness: {missing_pct:.1f}%")
        if duplicate_pct > 5:
            warnings.append(f"High duplicate rate: {duplicate_pct:.1f}%")
        if zero_var_cols:
            warnings.append(f"Zero-variance columns: {', '.join(zero_var_cols[:3])}")

        return TableFooter(
            row_count=len(df),
            col_count=len(df.columns),
            missing_pct=missing_pct,
            duplicate_pct=duplicate_pct,
            zero_variance_cols=zero_var_cols,
            warnings=warnings,
        )

    def _format_footer(self, footer: TableFooter) -> str:
        """Format footer as string."""
        lines = []
        lines.append("")
        lines.append("-" * 50)
        lines.append("[FOOTER]")
        lines.append(f"  Rows: {footer.row_count:,} | Columns: {footer.col_count}")
        lines.append(f"  Missing: {footer.missing_pct:.1f}% | Duplicates: {footer.duplicate_pct:.1f}%")

        if footer.zero_variance_cols:
            lines.append(f"  Zero-variance: {len(footer.zero_variance_cols)} column(s)")

        if footer.warnings:
            lines.append("  WARNINGS:")
            for w in footer.warnings:
                lines.append(f"    - {w}")

        lines.append("-" * 50)

        return "\n".join(lines)

    def get_printed_tables(self) -> List[Dict]:
        """Return all tables printed during this session (for export)."""
        return self._printed_tables

    def clear_history(self) -> None:
        """Clear the printed tables history."""
        self._printed_tables = []


# Convenience functions for quick use

_default_formatter = TableFormatter()


def print_table(
    df: pd.DataFrame,
    title: str,
    key_columns: Optional[List[str]] = None,
    show_footer: bool = True,
) -> None:
    """Print a formatted table to console."""
    output = _default_formatter.format_table(df, title, key_columns, show_footer)
    print(output)


def print_step_header(step_name: str, step_number: Optional[int] = None) -> None:
    """Print a large step header."""
    header = []
    header.append("")
    header.append("█" * 80)
    header.append("█" * 80)
    if step_number:
        header.append(f"██  STEP {step_number}: {step_name.upper()}")
    else:
        header.append(f"██  {step_name.upper()}")
    header.append("█" * 80)
    header.append("█" * 80)
    header.append("")
    print("\n".join(header))


def print_step_footer(
    step_name: str,
    elapsed_time: float,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Print a step summary footer."""
    footer = []
    footer.append("")
    footer.append("─" * 70)
    footer.append(f"✓ {step_name.upper()} COMPLETE")
    footer.append(f"  Time: {elapsed_time:.2f}s")

    if metrics:
        footer.append("  Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                footer.append(f"    {key}: {value:.4f}")
            else:
                footer.append(f"    {key}: {value}")

    footer.append("─" * 70)
    footer.append("")
    print("\n".join(footer))


def print_warning_block(
    warnings: List[str],
    title: str = "WARNINGS DETECTED",
    sample_df: Optional[pd.DataFrame] = None
) -> None:
    """Print a prominent warning block."""
    block = []
    block.append("")
    block.append("!" * 70)
    block.append(f"!  {title}")
    block.append("!" * 70)

    for i, w in enumerate(warnings, 1):
        block.append(f"  {i}. {w}")

    if sample_df is not None and len(sample_df) > 0:
        block.append("")
        block.append("  Sample of offending rows:")
        block.append(tabulate(
            sample_df.head(10),
            headers='keys',
            tablefmt='simple',
            showindex=False
        ))

    block.append("!" * 70)
    block.append("")
    print("\n".join(block))
