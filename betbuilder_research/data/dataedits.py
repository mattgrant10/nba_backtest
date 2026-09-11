#!/usr/bin/env python3
"""Compatibility launcher; the analysis lives in nba_research.

Use explicit --input and --output arguments. The original interactive code is
recoverable from preservation commit a85daa35b3650d3a41230d46f09976a106bb9110.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nba_research.__main__ import main

if __name__ == '__main__':
    raise SystemExit(main())
