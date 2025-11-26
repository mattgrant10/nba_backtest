#!/usr/bin/env python3
"""Verify the bet builder research setup."""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ❌ Python 3.9+ required")
        return False
    print("  ✓ Python version OK")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        "pandas", "numpy", "scipy", "sklearn",
        "joblib", "pyarrow", "tqdm"
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} not found")
            missing.append(package)

    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_directory_structure():
    """Check if directory structure exists."""
    required_dirs = [
        "data/raw",
        "data/processed",
        "models_artifacts",
        "src",
        "scripts"
    ]

    project_root = Path(__file__).parent
    missing = []

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ❌ {dir_path} not found")
            missing.append(dir_path)

    return len(missing) == 0

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "data/raw/player_game_logs.csv",
        "data/raw/player_prop_lines.csv",
        "data/raw/bet_builder_legs.csv"
    ]

    project_root = Path(__file__).parent
    missing = []

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ⚠ {file_path} not found (you need to provide this)")
            missing.append(file_path)

    if missing:
        print("\n  Note: Place your CSV files in data/raw/ directory")

    return True  # Don't fail on missing data files

def main():
    """Run all checks."""
    print("=" * 70)
    print("BET BUILDER RESEARCH - SETUP VERIFICATION")
    print("=" * 70)

    print("\n1. Checking Python version...")
    python_ok = check_python_version()

    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()

    print("\n3. Checking directory structure...")
    dirs_ok = check_directory_structure()

    print("\n4. Checking data files...")
    data_ok = check_data_files()

    print("\n" + "=" * 70)
    if python_ok and deps_ok and dirs_ok:
        print("✓ Setup verification passed!")
        print("\nNext steps:")
        print("1. Place your CSV files in data/raw/")
        print("2. Run: python scripts/prepare_data.py --validate --verbose")
        print("3. See QUICKSTART.md for the complete pipeline")
        return 0
    else:
        print("❌ Setup verification failed")
        print("\nPlease fix the issues above and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())
