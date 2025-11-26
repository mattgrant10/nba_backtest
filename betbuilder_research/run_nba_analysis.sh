#!/bin/bash
# Quick NBA Statistics Analysis
# Analyzes PlayerStatistics_2024_2025.csv with full visualizations

echo ""
echo "================================================================================"
echo "NBA PLAYER STATISTICS ANALYSIS - 2024-2025 SEASON"
echo "================================================================================"
echo ""
echo "Analyzing: data/raw/PlayerStatistics_2024_2025.csv"
echo ""
echo "This will perform:"
echo "  ✓ Player-level game-by-game analysis"
echo "  ✓ Season averages aggregation"
echo "  ✓ Team performance analysis"
echo "  ✓ Home vs away comparison"
echo "  ✓ Win/loss factors analysis"
echo "  ✓ Temporal trends"
echo "  ✓ Prop bet opportunity identification"
echo ""
echo "Estimated time: 2-3 minutes"
echo "Interactive plots will display automatically"
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if data file exists
if [ ! -f "data/raw/PlayerStatistics_2024_2025.csv" ]; then
    echo "❌ ERROR: data/raw/PlayerStatistics_2024_2025.csv not found"
    echo ""
    echo "Please ensure the file is in the correct location."
    exit 1
fi

echo "✓ Data file found"
echo ""
echo "Starting analysis..."
echo ""

# Run the analysis
python data/dataedits.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ ANALYSIS COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "DataFrames available for further analysis:"
    echo "  - df_clean: Cleaned game-level data"
    echo "  - player_agg: Player season averages"
    echo "  - team_agg: Team aggregated stats"
    echo "  - prop_analysis: Prop bet opportunities"
    echo ""
    echo "To continue analysis in Python:"
    echo "  python -i data/dataedits.py"
    echo ""
else
    echo ""
    echo "❌ Analysis failed. Check error messages above."
    exit 1
fi
