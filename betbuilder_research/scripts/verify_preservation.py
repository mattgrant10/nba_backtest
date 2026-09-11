#!/usr/bin/env python3
"""Compare complete legacy table exports with a completed refactored run."""

import argparse
import json
from pathlib import Path
import pandas as pd


def compare(baseline, result):
    names = sorted(p.name for p in Path(baseline).glob("*.csv"))
    if not names:
        raise ValueError("No baseline tables found")
    for name in names:
        old = pd.read_csv(Path(baseline) / name)
        new = pd.read_csv(Path(result) / "tables" / name)
        pd.testing.assert_frame_equal(old, new, check_exact=False, rtol=1e-10, atol=1e-12)
    return {
        "status": "passed",
        "tables_compared": len(names),
        "rtol": 1e-10,
        "atol": 1e-12,
        "tables": names,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.result), indent=2))
