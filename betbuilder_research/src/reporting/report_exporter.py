"""
Report Exporter Module.

Generates comprehensive HTML reports from pipeline runs.
Includes all tables, plots, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
import io
import json

import pandas as pd
import matplotlib.pyplot as plt

from ..config import get_logger, OUTPUT_DIR


logger = get_logger(__name__)


@dataclass
class RunMetadata:
    """Metadata about a pipeline run."""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    cli_args: Optional[Dict[str, Any]] = None
    python_version: Optional[str] = None
    backend: Optional[str] = None
    steps_completed: List[str] = None


class ReportExporter:
    """
    Exports pipeline results to HTML report.

    Collects tables, figures, warnings, and metrics throughout
    the pipeline run and generates a comprehensive report.
    """

    def __init__(self):
        self.tables: List[Dict] = []
        self.figures: List[Dict] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.step_results: Dict[str, Dict] = {}
        self.metadata: Optional[RunMetadata] = None

    def set_metadata(self, metadata: RunMetadata) -> None:
        """Set run metadata."""
        self.metadata = metadata

    def add_table(
        self,
        df: pd.DataFrame,
        title: str,
        step: str,
        description: Optional[str] = None
    ) -> None:
        """Add a table to the report."""
        self.tables.append({
            "df": df.copy(),
            "title": title,
            "step": step,
            "description": description,
        })

    def add_figure(
        self,
        fig: plt.Figure,
        title: str,
        step: str,
        description: Optional[str] = None
    ) -> None:
        """Add a figure to the report."""
        # Convert figure to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        self.figures.append({
            "img_base64": img_base64,
            "title": title,
            "step": step,
            "description": description,
        })

    def add_warning(self, warning: str, step: str) -> None:
        """Add a warning to the report."""
        self.warnings.append({
            "message": warning,
            "step": step,
        })

    def add_step_result(self, step: str, result: Dict[str, Any]) -> None:
        """Add step results."""
        self.step_results[step] = result

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set overall metrics."""
        self.metrics = metrics

    def generate_html(self) -> str:
        """Generate the full HTML report."""
        html_parts = []

        # Header
        html_parts.append(self._generate_header())

        # Metadata section
        html_parts.append(self._generate_metadata_section())

        # Warnings section
        if self.warnings:
            html_parts.append(self._generate_warnings_section())

        # Summary section
        html_parts.append(self._generate_summary_section())

        # Step-by-step results
        for step in self._get_ordered_steps():
            html_parts.append(self._generate_step_section(step))

        # Footer
        html_parts.append(self._generate_footer())

        return "\n".join(html_parts)

    def _generate_header(self) -> str:
        """Generate HTML header."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NBA Bet Builder Pipeline Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
            color: #333;
        }
        h1 { color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }
        h2 { color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 5px; margin-top: 40px; }
        h3 { color: #0f3460; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metadata { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
        .critical { background: #f8d7da; border-left: 4px solid #dc3545; }
        .metric-box { display: inline-block; background: #e8f4f8; padding: 15px; margin: 5px; border-radius: 5px; min-width: 150px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #16213e; }
        .metric-label { font-size: 12px; color: #666; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #16213e; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .figure { text-align: center; margin: 20px 0; }
        .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        .step-section { margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 5px; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }
    </style>
</head>
<body>
<div class="container">
<h1>NBA Bet Builder Pipeline Report</h1>
"""

    def _generate_metadata_section(self) -> str:
        """Generate metadata section."""
        if not self.metadata:
            return ""

        html = '<div class="metadata">'
        html += '<h3>Run Information</h3>'
        html += f'<p><strong>Run ID:</strong> {self.metadata.run_id}</p>'
        html += f'<p><strong>Start Time:</strong> {self.metadata.start_time.strftime("%Y-%m-%d %H:%M:%S")}</p>'

        if self.metadata.end_time:
            html += f'<p><strong>End Time:</strong> {self.metadata.end_time.strftime("%Y-%m-%d %H:%M:%S")}</p>'

        if self.metadata.total_duration:
            html += f'<p><strong>Duration:</strong> {self.metadata.total_duration:.2f}s</p>'

        if self.metadata.backend:
            html += f'<p><strong>Backend:</strong> {self.metadata.backend}</p>'

        if self.metadata.steps_completed:
            html += f'<p><strong>Steps:</strong> {", ".join(self.metadata.steps_completed)}</p>'

        html += '</div>'
        return html

    def _generate_warnings_section(self) -> str:
        """Generate warnings section."""
        html = '<h2>Warnings & Issues</h2>'

        for w in self.warnings:
            css_class = "warning critical" if "critical" in w.get("message", "").lower() else "warning"
            html += f'<div class="{css_class}">'
            html += f'<strong>[{w.get("step", "unknown")}]</strong> {w.get("message", "")}'
            html += '</div>'

        return html

    def _generate_summary_section(self) -> str:
        """Generate summary section with key metrics."""
        html = '<h2>Summary</h2>'
        html += '<div style="display: flex; flex-wrap: wrap;">'

        for key, value in self.metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            html += f'''
            <div class="metric-box">
                <div class="metric-value">{formatted_value}</div>
                <div class="metric-label">{key.replace("_", " ").title()}</div>
            </div>
            '''

        html += '</div>'
        return html

    def _generate_step_section(self, step: str) -> str:
        """Generate section for a single step."""
        html = f'<div class="step-section">'
        html += f'<h2>Step: {step.upper()}</h2>'

        # Step results
        if step in self.step_results:
            html += '<h3>Results</h3>'
            html += '<table>'
            for key, value in self.step_results[step].items():
                formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                html += f'<tr><td><strong>{key}</strong></td><td>{formatted}</td></tr>'
            html += '</table>'

        # Tables for this step
        step_tables = [t for t in self.tables if t.get("step") == step]
        for table in step_tables:
            html += f'<h3>{table["title"]}</h3>'
            if table.get("description"):
                html += f'<p>{table["description"]}</p>'
            html += table["df"].head(50).to_html(classes='dataframe', index=False)

        # Figures for this step
        step_figures = [f for f in self.figures if f.get("step") == step]
        for fig in step_figures:
            html += f'<div class="figure">'
            html += f'<h3>{fig["title"]}</h3>'
            html += f'<img src="data:image/png;base64,{fig["img_base64"]}" />'
            if fig.get("description"):
                html += f'<p>{fig["description"]}</p>'
            html += '</div>'

        html += '</div>'
        return html

    def _generate_footer(self) -> str:
        """Generate footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'''
<div class="footer">
    <p>Generated by NBA Bet Builder Pipeline</p>
    <p>Report created: {timestamp}</p>
</div>
</div>
</body>
</html>
'''

    def _get_ordered_steps(self) -> List[str]:
        """Get steps in execution order."""
        step_order = ["data", "distributions", "correlations", "monte-carlo", "edges", "train", "backtest"]
        all_steps = set()

        for t in self.tables:
            all_steps.add(t.get("step", ""))
        for f in self.figures:
            all_steps.add(f.get("step", ""))
        all_steps.update(self.step_results.keys())

        # Order by predefined order, then alphabetically for unknowns
        ordered = []
        for s in step_order:
            if s in all_steps:
                ordered.append(s)
                all_steps.remove(s)

        ordered.extend(sorted(all_steps))
        return ordered

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the report to file."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"pipeline_report_{timestamp}.html"

        path.parent.mkdir(parents=True, exist_ok=True)

        html = self.generate_html()
        with open(path, 'w') as f:
            f.write(html)

        logger.info(f"Report saved to: {path}")
        return path


def prompt_save_report(
    exporter: ReportExporter,
    default_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Prompt user to save report at end of pipeline.

    Returns path if saved, None otherwise.
    """
    try:
        response = input("\nSave report? (y/N): ").strip().lower()

        if response in ['y', 'yes']:
            path = exporter.save(default_path)
            print(f"\nReport saved to: {path}")
            return path
        else:
            print("Report not saved.")
            return None

    except (EOFError, KeyboardInterrupt):
        print("\nReport not saved.")
        return None
