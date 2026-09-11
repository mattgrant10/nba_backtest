"""Command line entry point. Configure headless rendering before importing plots."""

import argparse
import os
import tempfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run preserved NBA research on an explicit real CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/research"))
    parser.add_argument(
        "--mode",
        choices=["normal", "test"],
        default="normal",
        help="test retains verbose banners/tables and adds a log PDF; calculations are identical",
    )
    parser.add_argument(
        "--acca-template",
        type=Path,
        help="Optional explicit JSON specification; changes the replay specification",
    )
    args = parser.parse_args(argv)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nba-research-matplotlib")
    )
    from .pipeline import run

    try:
        import json

        template = json.loads(args.acca_template.read_text()) if args.acca_template else None
        run(args.input, args.output, args.mode, template)
    except Exception as error:
        parser.exit(1, f"Workflow failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
