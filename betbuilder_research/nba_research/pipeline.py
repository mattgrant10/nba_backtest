"""Complete real-data descriptive workflow, with explicit stage outputs."""

import json
import platform
import subprocess
import sys
import time
import importlib.metadata
from pathlib import Path
from .io import validate_input, export_tables, file_hash
from .logging import research_logging


def run(input_path, output_dir, mode="normal", acca_template=None):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    audit = validate_input(input_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir()
    console = sys.stdout
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(source_root)): file_hash(p)
        for folder in ("nba_research", "src")
        for p in sorted((source_root / folder).rglob("*.py"))
    }
    start = time.perf_counter()
    timings = {}
    results = {}
    from . import (
        preparation as prep,
        descriptive as desc,
        props,
        accumulator,
        distributions,
        dependence,
        plots,
    )
    from .plotting import figure_output
    from .report import write_report
    import seaborn as sns

    def stage(name, function, *args, **kwargs):
        print(f"[{len(timings) + 1:02d}] {name}", file=console, flush=True)
        begin = time.perf_counter()
        result = function(*args, **kwargs)
        timings[name] = round(time.perf_counter() - begin, 3)
        return result

    with (
        research_logging(output_dir / "analysis.log", mode == "test"),
        figure_output(figures),
        sns.axes_style("whitegrid"),
    ):
        raw = stage("Load data", prep.load_and_display_raw_data, input_path)
        clean = stage("Prepare data", prep.clean_and_prepare_data, raw)
        results["cleaned"] = clean
        stage("Player diagnostics", desc.analyze_player_level_stats, clean)
        results["players"] = stage("Player aggregates", desc.aggregate_by_player, clean)
        results["teams"] = stage("Team aggregates", desc.aggregate_by_team, clean)
        results["home_away"] = stage("Home and away", desc.analyze_home_away_performance, clean)
        results["win_loss"] = stage("Win and loss", desc.analyze_win_loss_performance, clean)
        results["daily_stats"] = stage("Temporal trends", desc.analyze_temporal_trends, clean)
        results["props"] = stage(
            "Player consistency", props.identify_prop_bet_opportunities, clean, results["players"]
        )
        results["hit_rates"] = stage(
            "Historical line hit rates",
            props.analyze_line_hit_rates,
            clean,
            results["props"],
            figures,
        )
        results["acca_daily"], results["acca_props"] = stage(
            "Fixed-template historical replay",
            accumulator.backtest_acca_across_dates,
            clean.copy(),
            results["props"],
            acca_template=acca_template,
        )
        # Preserve the original downstream date dtype following the historical replay.
        import pandas as pd

        clean["game_date_only"] = pd.to_datetime(clean["game_date_only"])
        results["distributions"] = stage(
            "Distribution diagnostics",
            distributions.test_distributions,
            clean,
            output_dir=figures,
            plot=True,
        )
        results["entropy"], results["mutual_information"], results["copula"] = stage(
            "Dependence diagnostics", dependence.analyze_advanced_dependencies, clean
        )
        stage("Distribution figures", plots.create_enhanced_seaborn_distributions, clean, figures)
        if not results["copula"].empty:
            stage(
                "Dependence figures",
                plots.create_copula_visualizations,
                clean,
                results["copula"],
                figures,
            )
            stage(
                "Rank and tail figures",
                plots.create_additional_copula_plots,
                clean,
                results["copula"],
                figures,
            )
            stage(
                "Illustrative family figures",
                plots.create_fitted_copula_plots,
                clean,
                results["copula"],
                figures,
            )
        stage(
            "Team consistency figures",
            plots.create_team_consistency_plots,
            clean,
            results["props"],
            figures,
        )
        tables = stage("Export tables", export_tables, results, output_dir / "tables")
        figure_paths = sorted(figures.rglob("*.png"))
        stage("Assemble report", write_report, output_dir / "report.pdf", figure_paths, audit)
    if mode == "test":
        write_report(
            output_dir / "detailed_log.pdf", [], audit, log_file=output_dir / "analysis.log"
        )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    packages = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "tabulate",
            "reportlab",
        ]
    }
    manifest = {
        "status": "complete",
        "mode": mode,
        "input": {"name": input_path.name, **audit},
        "acca_template": acca_template
        or json.loads((Path(__file__).parent / "acca_template.json").read_text()),
        "git_revision": revision,
        "source_sha256": source_hashes,
        "python": platform.python_version(),
        "packages": packages,
        "seconds": round(time.perf_counter() - start, 3),
        "stages_seconds": timings,
        "tables": tables,
        "figures": [str(p.relative_to(output_dir)) for p in figure_paths],
        "outputs": {
            str(p.relative_to(output_dir)): file_hash(p)
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        },
        "limitations": [
            "Descriptive, in-sample analysis; no validated betting profitability.",
            "Legacy copula and distribution diagnostic limitations: see docs/ANALYTICAL_REVIEW.md.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Complete: {len(tables)} tables, {len(figure_paths)} figures; {output_dir}", file=console
    )
    return manifest
