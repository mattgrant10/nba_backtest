# How to Run All Analysis - Complete Guide

## 🚀 Quick Start - Run Everything

```bash
cd /Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research

# Option 1: Run the comprehensive NBA data analysis (RECOMMENDED)
python data/dataedits.py

# Option 2: Run the full bet builder pipeline (requires all 3 CSV files)
./run_full_pipeline.sh
```

---

## 📋 Option 1: NBA Player Statistics Analysis (Current Season)

**What it does:** Analyzes your actual 2024-2025 NBA player statistics with comprehensive visualizations.

### Requirements
- File: `data/raw/PlayerStatistics_2024_2025.csv` ✓ (Already present)
- All dependencies installed ✓

### Run Command

```bash
python data/dataedits.py
```

### What You'll Get

**9 Analysis Steps with Interactive Visualizations:**

1. **Raw Data Loading** - Shows your 8,514 game records
2. **Data Cleaning** - Creates calculated fields, filters garbage time
3. **Player-Level Stats** - Game-by-game distributions and correlations
4. **Player Aggregation** - Season averages (PPG, APG, RPG)
5. **Team Aggregation** - Team rankings and performance
6. **Home vs Away** - Quantified home advantage
7. **Win vs Loss** - Factors that drive winning
8. **Temporal Trends** - Performance over time
9. **Prop Bet Opportunities** - Most consistent players identified

**Output:** 4 DataFrames saved to variables:
- `df_clean` - Cleaned game-level data
- `player_agg` - Player season averages
- `team_agg` - Team stats
- `prop_analysis` - Consistency metrics

**Time:** ~2-3 minutes with all visualizations

---

## 📋 Option 2: Full Bet Builder Research Pipeline

**What it does:** Complete pipeline from raw data to backtested betting strategies.

### Requirements

Place these 3 CSV files in `data/raw/`:

1. **player_game_logs.csv**
   - Columns: `game_id`, `game_date`, `player_id`, `pts`, `fg3m`, `ast`, `reb`, `home_away`

2. **player_prop_lines.csv**
   - Columns: `game_id`, `game_date`, `player_id`, `stat`, `line`, `over_odds`, `under_odds`

3. **bet_builder_legs.csv**
   - Columns: `builder_id`, `leg_id`, `game_id`, `game_date`, `player_id`, `stat`, `direction`, `line`, `builder_total_odds`, `stake`

### Step-by-Step Pipeline

#### Step 1: Prepare Data

```bash
python scripts/prepare_data.py --validate --verbose
```

**What it does:**
- Loads all 3 CSV files
- Validates data consistency
- Creates context features (home/away, rest days)
- Builds leg-level and builder-level datasets
- **Displays:** 15+ tabulated dataframes with plots

**Output files:**
- `data/processed/player_games.parquet`
- `data/processed/bet_builder_legs.parquet`
- `data/processed/bet_builders.parquet`

**Time:** ~5-10 minutes depending on data size

---

#### Step 2: Fit Player Distributions

```bash
python scripts/fit_player_distributions.py --compute-tail-probs --find-value-bets --verbose
```

**What it does:**
- Computes empirical distributions for each player-stat
- Fits Poisson/Negative Binomial models
- Calculates tail probabilities for prop lines
- Identifies value betting opportunities

**Output files:**
- `models_artifacts/player_stat_distributions.parquet`
- `models_artifacts/tail_probabilities.parquet`
- `models_artifacts/value_bets.parquet`

**Time:** ~2-5 minutes

---

#### Step 3: Analyze Dependencies

```bash
python scripts/analyze_dependencies.py --verbose
```

**What it does:**
- Computes per-player stat correlations (Spearman)
- Calculates global correlation matrices
- Identifies high-correlation pairs
- Analyzes patterns across players

**Output files:**
- `models_artifacts/player_stat_correlations.parquet`
- `models_artifacts/global_stat_correlations.parquet`
- `models_artifacts/high_correlations.parquet`

**Time:** ~3-5 minutes

---

#### Step 4: Train Bet Builder Model

```bash
python scripts/train_bet_builder_model.py --analyze-features --verbose
```

**What it does:**
- Trains logistic regression model
- Predicts builder success probability
- Analyzes feature importance
- Evaluates model performance

**Output files:**
- `models_artifacts/builder_hit_model.joblib`
- `models_artifacts/builder_hit_model_feature_importance.parquet`
- `models_artifacts/builder_hit_model_metrics.parquet`

**Time:** ~1-2 minutes

---

#### Step 5: Run Backtest

```bash
python scripts/run_backtest.py --save-predictions --analyze --verbose
```

**What it does:**
- Rolling time-window backtesting
- Trains on historical data, tests on future
- Computes ROI, profit, hit rates
- Analyzes time trends and stability

**Output files:**
- `data/processed/backtest_results.parquet`
- `data/processed/backtest_predictions.parquet`
- `data/processed/backtest_analysis.txt`

**Time:** ~10-20 minutes depending on data size

---

## 🎯 Automated Pipeline Script

Create this script to run everything:

```bash
#!/bin/bash
# run_full_pipeline.sh

set -e  # Exit on error

echo "=========================================="
echo "BET BUILDER RESEARCH - FULL PIPELINE"
echo "=========================================="

echo ""
echo "Step 1/5: Preparing data..."
python scripts/prepare_data.py --validate --verbose

echo ""
echo "Step 2/5: Fitting player distributions..."
python scripts/fit_player_distributions.py --compute-tail-probs --find-value-bets --verbose

echo ""
echo "Step 3/5: Analyzing dependencies..."
python scripts/analyze_dependencies.py --verbose

echo ""
echo "Step 4/5: Training bet builder model..."
python scripts/train_bet_builder_model.py --analyze-features --verbose

echo ""
echo "Step 5/5: Running backtest..."
python scripts/run_backtest.py --save-predictions --analyze --verbose

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Check these directories for results:"
echo "  - data/processed/     : Processed datasets"
echo "  - models_artifacts/   : Trained models"
echo "  - data/processed/plots: Visualization plots"
```

Save this as `run_full_pipeline.sh`, then:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## 📊 Understanding the Output

### Key Files to Check

**1. Backtest Results**
```bash
# View summary
python -c "import pandas as pd; df = pd.read_parquet('data/processed/backtest_results.parquet'); print(df.describe())"
```

**2. Feature Importance**
```bash
# See what drives outcomes
python -c "import pandas as pd; df = pd.read_parquet('models_artifacts/builder_hit_model_feature_importance.parquet'); print(df.head(10))"
```

**3. Value Bets**
```bash
# Find opportunities
python -c "import pandas as pd; df = pd.read_parquet('models_artifacts/value_bets.parquet'); print(df.sort_values('best_edge', ascending=False).head(20))"
```

---

## 🎨 Visualization Options

All scripts support these flags:

```bash
# Show plots interactively (default)
python scripts/prepare_data.py --show-plots

# Save plots to disk
python scripts/prepare_data.py --save-plots --plots-dir data/processed/plots

# Both show and save
python scripts/prepare_data.py --show-plots --save-plots

# Quiet mode (no plots)
python scripts/prepare_data.py --no-show-plots
```

---

## 🔍 Interactive Analysis

After running any script, you can continue in Python:

```python
import pandas as pd
import sys
sys.path.append('/Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research')

# Load processed data
df_clean = pd.read_parquet('data/processed/player_games.parquet')
builders = pd.read_parquet('data/processed/bet_builders.parquet')

# Your custom analysis here
print(df_clean.head())
print(builders.describe())
```

---

## ⚡ Quick Commands Cheat Sheet

```bash
# Full pipeline (if you have all 3 CSV files)
./run_full_pipeline.sh

# Just NBA stats analysis (current season)
python data/dataedits.py

# Verify setup
python verify_setup.py

# Check what data you have
ls -lh data/raw/

# Check results
ls -lh data/processed/
ls -lh models_artifacts/

# View logs
tail -f betbuilder_research.log  # if you enabled logging to file
```

---

## 🐛 Troubleshooting

**Problem:** Import errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Problem:** File not found errors
```bash
# Solution: Check your data files
ls data/raw/
# Make sure CSV files are present
```

**Problem:** Memory errors with large datasets
```bash
# Solution: Process in chunks or increase RAM
# Or filter data to smaller date range
```

**Problem:** Plots not showing
```bash
# Solution: Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
# Try: export MPLBACKEND=TkAgg (or Qt5Agg)
```

---

## 📈 Expected Results

### NBA Stats Analysis (dataedits.py)
- **Runtime:** 2-3 minutes
- **Plots:** ~15 interactive visualizations
- **Output:** 4 DataFrames with 600+ players analyzed

### Full Pipeline (all 5 scripts)
- **Runtime:** 30-45 minutes total
- **Plots:** ~50+ visualizations across all steps
- **Output:** Trained models, backtest results, performance metrics

---

## 🎯 Next Steps After Analysis

1. **Review top performers** - Check `player_agg` for season leaders
2. **Identify value bets** - Look at `value_bets.parquet` for opportunities
3. **Check model performance** - Review `backtest_results.parquet` for ROI
4. **Analyze consistency** - Use `prop_analysis` for reliable players
5. **Iterate** - Adjust features, windows, filters and re-run

---

## 📚 Additional Resources

- **Full Documentation:** `README.md`
- **Quick Start:** `QUICKSTART.md`
- **Setup Verification:** `python verify_setup.py`
- **Configuration:** Edit `src/config.py` for custom settings

---

## ✨ Tips for Best Results

1. **Start with NBA stats analysis** - Fastest way to see the framework in action
2. **Use verbose mode** - See everything that's happening: `--verbose`
3. **Save plots** - Create permanent record: `--save-plots`
4. **Check intermediate outputs** - Validate each step before continuing
5. **Customize configs** - Edit `src/config.py` for your preferences

---

**Ready to run? Start here:**

```bash
cd /Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research
python data/dataedits.py
```

This will analyze your actual 2024-2025 season data with full visualizations in ~2-3 minutes!
