#!/usr/bin/env python3
"""Re-run the preserved source with only the documented bounded-plot repair.

No alternative dataset is selected. Legacy source comes from local Git history.
"""

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a85daa35b3650d3a41230d46f09976a106bb9110"
RESULT_NAMES = [
    "cleaned",
    "players",
    "teams",
    "props",
    "hit_rates",
    "acca_daily",
    "acca_props",
    "distributions",
    "entropy",
    "mutual_information",
    "copula",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_input = args.input.resolve()
    output = args.output.resolve()
    if not source_input.is_file():
        raise FileNotFoundError(source_input)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    sys.path.insert(0, str(ROOT))
    source = subprocess.check_output(
        ["git", "show", f"{REVISION}:betbuilder_research/data/dataedits.py"], cwd=ROOT, text=True
    )
    old = "x_int = np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1)"
    new = "x_int = np.unique(np.linspace(int(np.floor(x_min)), int(np.ceil(x_max)), min(2000, int(np.ceil(x_max) - np.floor(x_min)) + 1)).astype(int))"
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace(
        'data_file = PROJECT_ROOT / "data/raw/PlayerStatistics_2025-2026_Feb.csv"',
        "data_file = INPUT_PATH",
    )
    module = types.ModuleType("preserved_legacy")
    module.__file__ = str(ROOT / "data/dataedits.py")
    module.INPUT_PATH = source_input
    os.chdir(output)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    results = module.main(detailed_log=False)
    from nba_research.io import export_tables

    export_tables(dict(zip(RESULT_NAMES, results[:11])), output / "tables")


if __name__ == "__main__":
    main()
