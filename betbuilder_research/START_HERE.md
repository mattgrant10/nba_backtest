# 🚀 START HERE - Quick Launch Guide

## ⚡ Fastest Way to See Results (2 minutes)

You have actual NBA data ready to analyze! Run this now:

```bash
./run_nba_analysis.sh
```

**OR:**

```bash
python data/dataedits.py
```

This will analyze your **PlayerStatistics_2024_2025.csv** file with:
- 8,514 game records
- 600+ players
- Full 2024-2025 season stats
- 15+ interactive visualizations
- Comprehensive prop bet opportunities

---

## 📊 What You'll Get

### Analysis Sections

1. **Player-Level Stats** (Game by Game)
   - Points, assists, rebounds distributions
   - Shooting percentages
   - Top performances
   - Correlation analysis

2. **Season Averages** (Player Aggregation)
   - PPG/APG/RPG leaders
   - Most efficient shooters
   - Consistency metrics

3. **Team Performance**
   - Win rate rankings
   - Team statistics
   - Performance comparison

4. **Home vs Away**
   - Quantified home advantage
   - Statistical differences

5. **Win vs Loss Factors**
   - What drives winning
   - Performance in wins/losses

6. **Temporal Trends**
   - Performance by day of week
   - Time series analysis

7. **🎯 Prop Bet Opportunities**
   - Most consistent scorers
   - Reliable assist players
   - Dependable rebounders

---

## 🎬 Three Ways to Run Analysis

### Option 1: NBA Stats Analysis (READY NOW ✅)

```bash
# Simple command
./run_nba_analysis.sh

# Or directly
python data/dataedits.py

# What you need: ✅ PlayerStatistics_2024_2025.csv (already present)
# Time: 2-3 minutes
# Output: 4 DataFrames + 15 plots
```

### Option 2: Full Bet Builder Pipeline (Requires 3 CSVs)

```bash
./run_full_pipeline.sh

# What you need:
#   - player_game_logs.csv
#   - player_prop_lines.csv  
#   - bet_builder_legs.csv
# Time: 30-45 minutes
# Output: Trained models + backtest results
```

### Option 3: Step-by-Step Pipeline

```bash
# 1. Prepare data
python scripts/prepare_data.py --validate --verbose

# 2. Fit distributions
python scripts/fit_player_distributions.py --compute-tail-probs --verbose

# 3. Analyze correlations
python scripts/analyze_dependencies.py --verbose

# 4. Train model
python scripts/train_bet_builder_model.py --analyze-features --verbose

# 5. Run backtest
python scripts/run_backtest.py --save-predictions --analyze --verbose
```

---

## 📁 Where Files Are Located

```
betbuilder_research/
├── data/
│   ├── raw/
│   │   └── PlayerStatistics_2024_2025.csv ✅ (YOUR DATA)
│   └── processed/  (output goes here)
│
├── models_artifacts/  (trained models go here)
│
├── data/dataedits.py ⭐ (NBA ANALYSIS SCRIPT)
│
├── scripts/  (full pipeline scripts)
│   ├── prepare_data.py
│   ├── fit_player_distributions.py
│   ├── analyze_dependencies.py
│   ├── train_bet_builder_model.py
│   └── run_backtest.py
│
└── Quick Launch Scripts:
    ├── run_nba_analysis.sh ⚡ (RUN THIS FIRST)
    └── run_full_pipeline.sh
```

---

## 🎯 Recommended First Steps

1. **Run NBA analysis** (you can do this right now!)
   ```bash
   ./run_nba_analysis.sh
   ```

2. **Review the output**
   - Check tabulated DataFrames in terminal
   - Interact with plots as they appear
   - Note the prop bet opportunities

3. **Explore results**
   - Top scorers by PPG
   - Most consistent players
   - Home advantage quantified
   - Team rankings

4. **Optional: Continue in Python**
   ```bash
   python -i data/dataedits.py
   # DataFrames are loaded and ready
   ```

---

## 📚 Full Documentation

- **This file** - Quick start (you're here!)
- **RUN_ANALYSIS.md** - Complete guide with all options
- **README.md** - Full project documentation
- **QUICKSTART.md** - Condensed setup guide

---

## 🐛 Troubleshooting

**"Command not found"**
```bash
# Make scripts executable
chmod +x run_nba_analysis.sh
chmod +x run_full_pipeline.sh
```

**"File not found"**
```bash
# Check you're in the right directory
pwd
# Should show: .../nba_backtest/betbuilder_research

# Check data file exists
ls -lh data/raw/PlayerStatistics_2024_2025.csv
```

**"Import errors"**
```bash
# Install dependencies
pip install -r requirements.txt
```

---

## ✨ What Makes This Analysis Special

✅ **Extremely detailed logging** - Every step explained
✅ **Professional visualizations** - Publication-quality plots
✅ **Tabulated output** - Clean, readable tables
✅ **Both player and aggregated analysis** - Complete picture
✅ **Prop bet identification** - Actionable insights
✅ **Interactive plots** - Explore your data visually
✅ **Production-ready code** - Error handling, validation

---

## 🎬 Ready? Let's Go!

```bash
cd /Users/matthewraymondandrewgrant/PycharmProjects/nba_backtest/betbuilder_research
./run_nba_analysis.sh
```

**Results in 2-3 minutes!** ⚡

---

Need help? Check **RUN_ANALYSIS.md** for the complete guide.
