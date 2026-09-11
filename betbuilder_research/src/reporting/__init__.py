"""
Reporting Module for NBA Bet Builder Pipeline.

Console-first, research-grade output with structured tables,
live visual feedback, and optional report export.
"""

from .table_formatter import (
    TableFormatter,
    print_table,
    print_step_header,
    print_step_footer,
    print_warning_block,
)

from .diagnostics import (
    DataDiagnostics,
    run_data_health_checks,
    check_zero_variance,
    check_duplicates,
    check_outliers,
)

from .visuals import (
    VisualManager,
    show_distribution,
    show_heatmap,
    show_histogram,
)

from .report_exporter import (
    ReportExporter,
    RunMetadata,
    prompt_save_report,
)

from .pdf_exporter import (
    PDFReportExporter,
    PDFReportConfig,
    ConsoleCapture,
    create_tiered_tables,
    get_tier_description,
    DATA_FEED_DOCUMENTATION,
)

__all__ = [
    # Table formatting
    "TableFormatter",
    "print_table",
    "print_step_header",
    "print_step_footer",
    "print_warning_block",
    # Diagnostics
    "DataDiagnostics",
    "run_data_health_checks",
    "check_zero_variance",
    "check_duplicates",
    "check_outliers",
    # Visuals
    "VisualManager",
    "show_distribution",
    "show_heatmap",
    "show_histogram",
    # Export
    "ReportExporter",
    "RunMetadata",
    "prompt_save_report",
    # PDF Export
    "PDFReportExporter",
    "PDFReportConfig",
    "ConsoleCapture",
    "create_tiered_tables",
    "get_tier_description",
    "DATA_FEED_DOCUMENTATION",
]
