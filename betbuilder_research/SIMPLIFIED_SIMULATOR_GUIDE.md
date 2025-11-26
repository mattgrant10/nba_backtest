# Simplified Accumulator Simulator Guide

## Overview

The new `simulate_acca()` function provides a **simple, streamlined way** to test accumulator bets without any complex setup.

**No more:**
- ❌ Game details (date, time, match, odds)
- ❌ Leg structures
- ❌ Date ranges
- ❌ Complex templates

**Just need:**
- ✅ Player props (player, stat, line, over/under)
- ✅ Number of games to simulate

---

## 🚀 Quick Start

### Interactive Mode (Easiest)

After running the main analysis, simply call:

```python
results, props_df = simulate_acca(df_clean, prop_analysis)
```

You'll be asked to input:
1. **Player names** (with search and validation)
2. **Stat type** (PTS, AST, REB, BLK, STL, FG3M, TOV, MIN)
3. **Line value** (e.g., 20.5)
4. **Direction** (Over or Under)
5. **Number of games** to simulate

### Example Interactive Session

```
================================================================================
ACCUMULATOR SIMULATOR
================================================================================

Add your player props one by one.
Available stats: PTS, AST, REB, BLK, STL, FG3M, TOV, MIN
================================================================================

================================================================================
PROP 1
================================================================================

Enter player name (or 'search' to find players, 'done' to finish): LaMelo Ball
✓ Found: LaMelo Ball

📊 LaMelo Ball's Season Averages:
    PPG: 31.1, APG: 6.9, RPG: 5.4
    Games: 16

Enter stat (PTS/AST/REB/BLK/STL/FG3M/TOV/MIN): PTS
Enter line for PTS: 20.5
Over or Under? (O/U): O

✓ Added: LaMelo Ball Over 20.5 PTS

Add another prop? (Y/N): Y

================================================================================
PROP 2
================================================================================

Enter player name (or 'search' to find players, 'done' to finish): Bridges
⚠️  Multiple players found:
  1. Mikal Bridges
  2. Miles Bridges

Enter number to select: 1
✓ Selected: Mikal Bridges

Enter stat (PTS/AST/REB/BLK/STL/FG3M/TOV/MIN): PTS
Enter line for PTS: 18.5
Over or Under? (O/U): O

✓ Added: Mikal Bridges Over 18.5 PTS

Add another prop? (Y/N): N

================================================================================
SIMULATION SETTINGS
================================================================================

Total game dates available in dataset: 63

How many games to simulate? (1-63): 10

================================================================================
ACCUMULATOR SUMMARY
================================================================================
1. LaMelo Ball - Over 20.5 PTS
2. Mikal Bridges - Over 18.5 PTS

Total props: 2
Games to simulate: 10
================================================================================

Run simulation? (Y/N): Y
```

---

## 💻 Programmatic Mode (For Scripts)

### Basic Usage

```python
# Define your props
props = [
    {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 20.5, 'over': True},
    {'player': 'Mikal Bridges', 'stat': 'PTS', 'line': 18.5, 'over': True},
    {'player': 'Franz Wagner', 'stat': 'REB', 'line': 5.5, 'over': True}
]

# Run simulation for 20 games
results, props_df = simulate_acca(
    df_clean,
    prop_analysis,
    props=props,
    n_games=20
)
```

### Prop Structure

Each prop is a dictionary with these keys:

```python
{
    'player': 'Player Name',     # Full or partial name (will be matched)
    'stat': 'PTS',               # PTS, AST, REB, BLK, STL, FG3M, TOV, MIN
    'line': 20.5,                # Numeric line value
    'over': True                 # True for Over, False for Under
}
```

### Available Stats

| Stat | Description |
|------|-------------|
| PTS | Points |
| AST | Assists |
| REB | Rebounds |
| BLK | Blocks |
| STL | Steals |
| FG3M | Three-Pointers Made |
| TOV | Turnovers |
| MIN | Minutes Played |

---

## 📊 Understanding Results

### Results DataFrame

Contains game-by-game accumulator results:

```python
>>> results
   game_num        date  props_hit  props_tested  hit_rate  acca_hit
0         1  2024-11-15          2             2     100.0      True
1         2  2024-11-18          1             2      50.0     False
2         3  2024-11-22          2             2     100.0      True
...
```

**Columns:**
- `game_num`: Simulation game number (1, 2, 3, ...)
- `date`: Date of the simulated game
- `props_hit`: Number of props that hit
- `props_tested`: Number of props tested (some may DNP)
- `hit_rate`: Percentage of props that hit
- `acca_hit`: Whether the entire accumulator won (all props hit)

### Props DataFrame

Contains individual prop performance across all simulated games:

```python
>>> props_df
   game_num        date         player stat  line   over  result    hit  margin  player_cv
0         1  2024-11-15  LaMelo Ball  PTS  20.5   True    34.0   True    13.5      0.234
1         1  2024-11-15 Mikal Bridges  PTS  18.5   True    21.0   True     2.5      0.189
2         2  2024-11-18  LaMelo Ball  PTS  20.5   True    18.0  False    -2.5      0.234
...
```

**Columns:**
- `game_num`: Which simulation game
- `date`: Date of game
- `player`: Player name
- `stat`: Stat type
- `line`: Prop line
- `over`: True/False
- `result`: Actual stat value in that game
- `hit`: Whether prop hit (True/False)
- `margin`: Result - Line (positive = over, negative = under)
- `player_cv`: Player's coefficient of variation (consistency metric)

---

## 📈 Summary Statistics

After simulation completes, you'll see detailed logging:

```
====================================================================================================
SIMULATION RESULTS SUMMARY
====================================================================================================

>>> OVERALL PERFORMANCE:
    Games simulated: 10
    Wins: 6
    Losses: 4
    Win rate: 60.0%
    Average prop hit rate: 75.5%

    GAME-BY-GAME RESULTS:
┌────────────┬──────────────┬─────────────┬────────────────┬────────────┬────────────┐
│ game_num   │ date         │ props_hit   │ props_tested   │ hit_rate   │ acca_hit   │
├────────────┼──────────────┼─────────────┼────────────────┼────────────┼────────────┤
│ 1          │ 2024-11-15   │ 2           │ 2              │ 100.0      │ True       │
│ 2          │ 2024-11-18   │ 1           │ 2              │ 50.0       │ False      │
│ 3          │ 2024-11-22   │ 2           │ 2              │ 100.0      │ True       │
...
└────────────┴──────────────┴─────────────┴────────────────┴────────────┴────────────┘

>>> PROP PERFORMANCE:
    LaMelo Ball PTS O20.5:
        Hit rate: 70.0% (7/10)
        Avg margin: +5.25
        CV: 0.234
    Mikal Bridges PTS O18.5:
        Hit rate: 80.0% (8/10)
        Avg margin: +3.15
        CV: 0.189
```

---

## 🎯 Key Features

### 1. **Random Game Sampling**
- Randomly selects N games from the dataset
- Ensures diverse testing across different dates
- Uses seed=42 for reproducibility

### 2. **Player DNP Handling**
- Automatically skips props where player didn't play
- Game is void if all props are DNP
- Logs which players were skipped each game

### 3. **Detailed Logging**
Every simulation game shows:
```
>>> Game 5/10: 2024-11-22
    12 games played that night
        LaMelo Ball PTS O20.5: 34.0 - HIT ✓ (margin: +13.5)
        Mikal Bridges PTS O18.5: 21.0 - HIT ✓ (margin: +2.5)
    RESULT: WIN ✓✓✓ (2/2 props hit = 100.0%)
```

### 4. **Player Consistency Metrics**
- Each prop result includes player's CV (coefficient of variation)
- Lower CV = more consistent player
- Helps identify which props are more reliable

### 5. **Margin Analysis**
- Shows by how much each prop hit or missed
- Positive margin = result exceeded line
- Negative margin = result fell short

---

## 🔧 Advanced Usage

### Test Multiple Scenarios

```python
# Scenario 1: Conservative props
props_conservative = [
    {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 15.5, 'over': True},
    {'player': 'Mikal Bridges', 'stat': 'PTS', 'line': 12.5, 'over': True}
]
results1, props1 = simulate_acca(df_clean, prop_analysis, props=props_conservative, n_games=50)

# Scenario 2: Aggressive props
props_aggressive = [
    {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 30.5, 'over': True},
    {'player': 'Mikal Bridges', 'stat': 'PTS', 'line': 22.5, 'over': True}
]
results2, props2 = simulate_acca(df_clean, prop_analysis, props=props_aggressive, n_games=50)

# Compare win rates
print(f"Conservative win rate: {results1['acca_hit'].mean()*100:.1f}%")
print(f"Aggressive win rate: {results2['acca_hit'].mean()*100:.1f}%")
```

### Filter Results

```python
# Get only winning games
wins = results[results['acca_hit'] == True]

# Get games where all props hit with big margins
props_df['big_margin'] = props_df['margin'].abs() > 5
big_wins = props_df[props_df['big_margin']]

# Check consistency of losses
losses = results[results['acca_hit'] == False]
print(f"Average props hit in losses: {losses['props_hit'].mean():.1f}")
```

### Analyze Individual Props

```python
# Which prop is most reliable?
for prop in props:
    prop_results = props_df[props_df['player'].str.contains(prop['player'], case=False)]
    prop_results = prop_results[prop_results['stat'] == prop['stat']]

    hit_rate = prop_results['hit'].mean() * 100
    avg_margin = prop_results['margin'].mean()

    print(f"{prop['player']} {prop['stat']}: {hit_rate:.1f}% hit rate, avg margin: {avg_margin:+.2f}")
```

---

## 📍 Location in Code

**File:** `data/dataedits.py`
**Function:** `simulate_acca()` (lines 936-1290)

**To use after running main analysis:**
```python
python -i data/dataedits.py

# Then:
>>> results, props_df = simulate_acca(df_clean, prop_analysis)
```

---

## 🆚 Comparison with Other Methods

| Feature | `simulate_acca()` | `interactive_acca_builder()` | `backtest_acca_across_dates()` |
|---------|------------------|----------------------------|------------------------------|
| Input complexity | ⭐ Simple | ⭐⭐ Moderate | ⭐⭐⭐ Complex |
| Requires legs | ❌ No | ✅ Yes | ✅ Yes |
| Requires dates | ❌ No | ✅ Yes | ✅ Yes |
| Requires game details | ❌ No | ❌ No | ✅ Yes |
| Random sampling | ✅ Yes | ❌ No (all dates) | ❌ No (specific range) |
| Best for | Quick tests | Structured bets | Full backtesting |

---

## ❓ FAQ

**Q: What if player name is ambiguous?**
A: The function will show you all matches and let you select by number.

**Q: What happens if a player doesn't play?**
A: The prop is automatically skipped for that game. If all props are skipped, the game is voided.

**Q: Can I test the same prop across all dates?**
A: Yes, just set `n_games` to the total number of dates available.

**Q: How are games selected?**
A: Random sampling with seed=42 (reproducible). If you want 10 games, it randomly picks 10 dates from the dataset.

**Q: Can I use partial player names?**
A: Yes! "Ball" will match "LaMelo Ball", "Bridges" will match both Bridges brothers (you'll be asked to choose).

**Q: What stats can I use?**
A: PTS, AST, REB, BLK, STL, FG3M, TOV, MIN

---

## 💡 Tips

1. **Start with fewer games** (5-10) to quickly test your props
2. **Check player consistency** (CV values) - lower is better
3. **Look at margins** - props that barely hit/miss are risky
4. **Test multiple scenarios** - conservative vs aggressive lines
5. **Use programmatic mode** for batch testing multiple prop combinations

---

## ✅ Summary

**Simplest way to test accumulator bets:**

```python
results, props_df = simulate_acca(df_clean, prop_analysis)
```

**Just need:**
- Player names
- Stats (PTS/AST/REB/etc.)
- Lines
- Over/Under
- Number of games

**Get back:**
- Win/loss for each simulated game
- Individual prop performance
- Detailed statistics and margins
- Player consistency metrics

**No more complex templates, legs, dates, or game details needed!**
