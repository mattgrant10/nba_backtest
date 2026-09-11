"""Optional local integration checks; never download or generate replacement data."""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nba_research.io import validate_input, file_hash

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.environ.get("NBA_TEST_INPUT", ROOT / "data/raw/PlayerStatistics_2025-2026_Feb.csv"))
OUTPUT = Path(os.environ.get("NBA_TEST_OUTPUT", ROOT / "output/research"))
pytestmark = pytest.mark.real_data


@pytest.fixture
def real_data():
    if not INPUT.exists():
        pytest.skip("Local real CSV not supplied")
    return pd.read_csv(INPUT)


def test_source_fitness(real_data):
    audit = validate_input(INPUT)
    expected = json.loads((ROOT / "tests/baseline_manifest.json").read_text())
    assert audit["sha256"] == expected["input_sha256"]
    assert len(real_data) == 24092
    assert (real_data.numMinutes >= 5).sum() == 18346


def test_duplicate_keys_rejected(real_data, tmp_path):
    path = tmp_path / "duplicate.csv"
    pd.concat([real_data, real_data.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_input(path)


def test_invalid_boolean_rejected(real_data, tmp_path):
    real_data["home"] = real_data.home.astype(object)
    real_data.loc[0, "home"] = "False"
    path = tmp_path / "bad_boolean.csv"
    real_data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="home must"):
        validate_input(path)


def test_completed_outputs_and_all_baseline_tables(real_data):
    if not (OUTPUT / "manifest.json").exists():
        pytest.skip("Run the complete workflow first")
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    expected = json.loads((ROOT / "tests/baseline_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["figures"]) == 317
    assert manifest["input"]["sha256"] == expected["input_sha256"]
    for name, item in expected["tables"].items():
        # Exact hashes supplement tolerance-based comparison against full legacy tables.
        reference = ROOT / "tests/reference" / item["file"]
        if reference.exists():
            pd.testing.assert_frame_equal(
                pd.read_csv(reference),
                pd.read_csv(OUTPUT / "tables" / item["file"]),
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        else:
            assert manifest["tables"][name]["sha256"] == item["sha256"], name
    for path, digest in manifest["outputs"].items():
        assert file_hash(OUTPUT / path) == digest, path
    from pypdf import PdfReader

    assert len(PdfReader(OUTPUT / "report.pdf").pages) == 1 + len(manifest["figures"])
    # Independent raw-data check on the retained game-level distribution.
    cleaned = pd.read_csv(OUTPUT / "tables/cleaned.csv")
    np.testing.assert_array_equal(cleaned.PTS, real_data.loc[real_data.numMinutes >= 5, "points"])
    assert len(cleaned) == 18346
