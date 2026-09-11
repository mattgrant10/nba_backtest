"""Input contracts and portable table/provenance exports."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = [
    "personId",
    "gameId",
    "gameDateTimeEst",
    "firstName",
    "lastName",
    "playerteamCity",
    "playerteamName",
    "opponentteamCity",
    "opponentteamName",
    "win",
    "home",
    "numMinutes",
    "points",
    "assists",
    "reboundsTotal",
    "threePointersMade",
    "steals",
    "blocks",
    "turnovers",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersAttempted",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "plusMinusPoints",
]
NUMERIC = REQUIRED[11:]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_input(path):
    """Reject ambiguous/invalid inputs; do not impute, deduplicate or replace data."""
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED) - set(df))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input contains no player-games")
    nullable = {"playerteamCity", "opponentteamCity", "win", "numMinutes"}
    if df[[c for c in REQUIRED if c not in nullable]].isna().any().any():
        raise ValueError("Missing values in required input columns")
    dates = pd.to_datetime(df["gameDateTimeEst"], format="mixed", errors="raise")
    if df.duplicated(["personId", "gameId"]).any():
        raise ValueError("Duplicate personId/gameId records; resolve upstream")
    for column in ["win", "home"]:
        if not df[column].dropna().isin([0, 1, True, False]).all():
            raise ValueError(f"{column} must contain boolean or 0/1 values")
    for column in NUMERIC:
        if (
            not pd.api.types.is_numeric_dtype(df[column])
            or not np.isfinite(df[column].dropna()).all()
        ):
            raise ValueError(f"{column} must contain finite numeric values")
        if column != "plusMinusPoints" and (df[column] < 0).any():
            raise ValueError(f"{column} contains negative values")
    if not (df.numMinutes >= 5).any():
        raise ValueError("No player-games meet the preserved five-minute filter")
    return {
        "rows": len(df),
        "players": int(df.personId.nunique()),
        "games": int(df.gameId.nunique()),
        "date_start": str(dates.min()),
        "date_end": str(dates.max()),
        "sha256": file_hash(path),
        "excluded_under_five_minutes": int((~(df.numMinutes >= 5)).sum()),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
        "warnings": [
            "Legacy missing wins convert to True before filtering (zero missing wins retained in the reference dataset); missing team cities exclude rows from some grouped summaries. No imputation applied."
        ],
    }


def export_tables(results, directory):
    """Export every returned DataFrame, including nested hit-rate tables."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {}

    def visit(name, value):
        if isinstance(value, pd.DataFrame):
            path = directory / f"{name}.csv"
            value.to_csv(path, index=True)
            tables[name] = {
                "rows": len(value),
                "columns": len(value.columns),
                "file": path.name,
                "sha256": file_hash(path),
            }
        elif isinstance(value, dict):
            for key, item in value.items():
                if key != "figures":
                    visit(f"{name}__{key}", item)

    for name, value in results.items():
        visit(name, value)
    return tables
