# Quick Start Guide

## 1. Setup (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data

Place three CSV files in `data/raw/`:
- `player_game_logs.csv`
- `player_prop_lines.csv`
- `bet_builder_legs.csv`

See README.md for required columns.

## 3. Run the Pipeline (in order)

```bash
# Step 1: Prepare and validate data
python scripts/prepare_data.py --validate --verbose

# Step 2: Fit player stat distributions
python scripts/fit_player_distributions.py --compute-tail-probs --verbose

# Step 3: Analyze stat correlations
python scripts/analyze_dependencies.py --verbose

# Step 4: Train bet builder model
python scripts/train_bet_builder_model.py --verbose

# Step 5: Run backtest
python scripts/run_backtest.py --save-predictions --analyze --verbose
```

## 4. Check Results

After running the pipeline, check:

- `data/processed/backtest_results.parquet` - Performance by time period
- `data/processed/backtest_analysis.txt` - Summary statistics
- `models_artifacts/builder_hit_model_feature_importance.parquet` - What drives outcomes

## 5. Interpret Results

Key metrics to look at:
- **Overall ROI**: Is it positive?
- **Hit Rate**: How often do predictions win?
- **Brier Score**: Are probabilities calibrated? (< 0.25 is good)
- **Win Rate**: What % of time periods are profitable?

## Common Issues

**Problem**: "File not found"
**Solution**: Make sure CSV files are in `data/raw/` directory

**Problem**: "Missing columns"
**Solution**: Check your CSV files have all required columns (see README.md)

**Problem**: Scripts fail with import errors
**Solution**: Make sure you're in the project root directory when running scripts

## Next Steps

Once the basic pipeline works:

1. Experiment with different features (edit `src/config.py`)
2. Adjust training windows for more/less historical data
3. Add custom features in `src/feature_engineering.py`
4. Try different models in `src/models/bet_builder_outcome.py`

## Getting Help

- Check README.md for detailed documentation
- Review log files for error details
- Ensure data quality with `--validate` flag

## Example Output

```
BACKTEST COMPLETE
================================================================================
Total folds completed: 20
Average ROI: 5.23%
Total profit: 45.67
Total stake: 873.00
Overall ROI: 5.23%
Profitable folds: 14/20 (70.0%)
```

This means:
- Tested on 20 different time periods
- Average return of 5.23% per period
- Made $45.67 profit on $873 total stakes
- 70% of periods were profitable

Remember: Past performance does not guarantee future results!
