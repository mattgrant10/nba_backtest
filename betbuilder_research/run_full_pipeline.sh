#!/bin/bash
# Full Pipeline Automation Script
# Runs all bet builder research analysis steps in sequence

set -e  # Exit on any error

echo ""
echo "================================================================================"
echo "BET BUILDER RESEARCH - FULL PIPELINE"
echo "================================================================================"
echo ""
echo "This script will run all 5 analysis steps:"
echo "  1. Data Preparation"
echo "  2. Player Distribution Fitting"
echo "  3. Dependency Analysis"
echo "  4. Model Training"
echo "  5. Backtesting"
echo ""
echo "Estimated total time: 30-45 minutes"
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if data files exist
echo ">>> Checking for required data files..."
if [ ! -f "data/raw/player_game_logs.csv" ]; then
    echo "❌ ERROR: data/raw/player_game_logs.csv not found"
    echo ""
    echo "To run the full pipeline, you need 3 CSV files in data/raw/:"
    echo "  - player_game_logs.csv"
    echo "  - player_prop_lines.csv"
    echo "  - bet_builder_legs.csv"
    echo ""
    echo "Alternatively, run the NBA stats analysis on existing data:"
    echo "  python data/dataedits.py"
    exit 1
fi

if [ ! -f "data/raw/player_prop_lines.csv" ]; then
    echo "❌ ERROR: data/raw/player_prop_lines.csv not found"
    exit 1
fi

if [ ! -f "data/raw/bet_builder_legs.csv" ]; then
    echo "❌ ERROR: data/raw/bet_builder_legs.csv not found"
    exit 1
fi

echo "✓ All required data files found"
echo ""

# Function to print step header
print_step() {
    echo ""
    echo "================================================================================"
    echo "STEP $1/5: $2"
    echo "================================================================================"
    echo ""
}

# Step 1: Prepare Data
print_step 1 "DATA PREPARATION"
echo ">>> Running: python scripts/prepare_data.py --validate --verbose"
python scripts/prepare_data.py --validate --verbose

# Step 2: Fit Distributions
print_step 2 "FIT PLAYER DISTRIBUTIONS"
echo ">>> Running: python scripts/fit_player_distributions.py"
python scripts/fit_player_distributions.py --compute-tail-probs --find-value-bets --verbose

# Step 3: Analyze Dependencies
print_step 3 "ANALYZE STAT DEPENDENCIES"
echo ">>> Running: python scripts/analyze_dependencies.py"
python scripts/analyze_dependencies.py --verbose

# Step 4: Train Model
print_step 4 "TRAIN BET BUILDER MODEL"
echo ">>> Running: python scripts/train_bet_builder_model.py"
python scripts/train_bet_builder_model.py --analyze-features --verbose

# Step 5: Run Backtest
print_step 5 "RUN BACKTEST"
echo ">>> Running: python scripts/run_backtest.py"
python scripts/run_backtest.py --save-predictions --analyze --verbose

# Success message
echo ""
echo "================================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  📁 data/processed/     - Processed datasets"
echo "  📁 models_artifacts/   - Trained models"
echo "  📁 data/processed/plots - Visualization plots"
echo ""
echo "Key files to check:"
echo "  📊 data/processed/backtest_results.parquet"
echo "  📊 models_artifacts/builder_hit_model.joblib"
echo "  📊 models_artifacts/value_bets.parquet"
echo ""
echo "To view results:"
echo "  python -c \"import pandas as pd; print(pd.read_parquet('data/processed/backtest_results.parquet'))\""
echo ""
echo "================================================================================"
