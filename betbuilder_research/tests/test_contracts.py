"""Boundary contracts use minimal test fixtures; integration tests use actual data."""

import logging
import sys
import pandas as pd
import pytest
from nba_research.io import validate_input
from nba_research.logging import research_logging
from src.data_loader import (
    DataLoadError,
    load_prop_lines,
    load_bet_builder_legs,
    load_player_game_logs,
)
from src.models.bet_builder_outcome import train_builder_hit_model


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_input(tmp_path / "absent.csv")


def test_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"personId": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(path)


@pytest.mark.parametrize("loader", [load_prop_lines, load_bet_builder_legs, load_player_game_logs])
def test_explicit_missing_file_never_falls_back(tmp_path, loader):
    with pytest.raises(DataLoadError):
        loader(tmp_path / "missing.csv")


@pytest.mark.parametrize("feature", ["mean_abs_edge", "max_abs_edge", "realised", "builder_hit"])
def test_known_leakage_rejected_before_training(feature):
    with pytest.raises(ValueError, match="Outcome-derived"):
        train_builder_hit_model(pd.DataFrame(), feature_cols=[feature])


@pytest.mark.parametrize("verbose", [False, True])
def test_logging_restores_streams_on_failure(tmp_path, verbose):
    original = sys.stdout
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    path = tmp_path / "analysis.log"
    with pytest.raises(RuntimeError):
        with research_logging(path, verbose):
            print("unique table message")
            logging.info("unique logger message")
            raise RuntimeError("stage failed")
    assert sys.stdout is original
    assert root.handlers == handlers and root.level == level
    content = path.read_text()
    assert content.count("unique table message") == 1
    assert content.count("unique logger message") == 1
