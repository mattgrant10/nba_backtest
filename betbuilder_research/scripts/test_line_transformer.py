#!/usr/bin/env python3
"""
Test the wide-format line transformer with sample data.

This script:
1. Loads the sample wide-format betting lines CSV
2. Transforms to long format with decimal odds
3. Displays the result
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_logging, RAW_DATA_DIR
from src.models.line_generator import (
    load_wide_format_lines,
    american_to_decimal,
)


def main():
    setup_logging()

    print("=" * 70)
    print("TESTING WIDE-FORMAT LINE TRANSFORMER")
    print("=" * 70)

    # Test American to Decimal conversion
    print("\n1. Testing American to Decimal odds conversion:")
    print("-" * 50)
    test_odds = [-106, -110, -117, +100, +104, +120, -145, -125]
    for odds in test_odds:
        decimal = american_to_decimal(odds)
        print(f"   {odds:+4d} -> {decimal:.3f}")

    # Load sample data
    sample_path = RAW_DATA_DIR / "sample_betting_lines.csv"

    if not sample_path.exists():
        print(f"\nError: Sample file not found: {sample_path}")
        return 1

    print(f"\n2. Loading and transforming: {sample_path}")
    print("-" * 50)

    long_df = load_wide_format_lines(sample_path, game_date="2025-12-26")

    print(f"\n3. Transformed Data Summary:")
    print("-" * 50)
    print(f"   Total lines: {len(long_df)}")
    print(f"   Unique players: {long_df['player_name'].nunique()}")
    print(f"   Stats breakdown:")
    for stat, count in long_df['stat'].value_counts().items():
        print(f"      {stat}: {count} lines")

    print(f"\n4. Sample Output (first 15 rows):")
    print("-" * 50)

    # Display sample
    display_cols = ['player_name', 'team', 'stat', 'line', 'over_odds', 'under_odds']
    print(long_df[display_cols].head(15).to_string(index=False))

    print(f"\n5. Odds Range Check (should be ~1.8-2.3 for decimal):")
    print("-" * 50)
    print(f"   Over odds:  min={long_df['over_odds'].min():.3f}, max={long_df['over_odds'].max():.3f}")
    print(f"   Under odds: min={long_df['under_odds'].min():.3f}, max={long_df['under_odds'].max():.3f}")

    print("\n" + "=" * 70)
    print("TRANSFORMATION SUCCESSFUL")
    print("=" * 70)

    # Save transformed data for reference
    output_path = RAW_DATA_DIR / "sample_betting_lines_long.csv"
    long_df.to_csv(output_path, index=False)
    print(f"\nTransformed data saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
