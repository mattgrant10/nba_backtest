# Line Hit Rate Analysis - Guide (TOP PLAYERS)

## Overview

The **Line Hit Rate Analysis** calculates the success percentage (hit rate) for **TOP PLAYERS** at different line positions relative to their season averages. This focuses on the league leaders who are most commonly featured in prop bets.

---

## 🎯 What is Hit Rate?

**Hit Rate Definition:**
> The percentage of games where a player's actual result was **OVER** the specified line.

**Example:**
- Player averages 20.0 PPG over 20 games
- In 12 games, they scored more than 20 points
- **Hit Rate at Average = 60%** (12/20 games)

---

## 🏀 Players Analyzed

The analysis focuses on **league leaders only**:

1. **Top 30 Scorers** - Highest PPG (Points Per Game)
2. **Top 30 Assist Leaders** - Highest APG (Assists Per Game)
3. **Top 30 Rebounders** - Highest RPG (Rebounds Per Game)

These are the players most commonly featured in prop betting markets.

---

## 📊 Line Positions Analyzed

For each top player, we calculate hit rates at **4 different line positions**:

| Line Position | Offset from Average | Expected Hit Rate | Use Case |
|---------------|-------------------|-------------------|-----------|
| **Avg - 10** | Average minus 10 | ~80-90% | Very conservative lines (high confidence) |
| **Avg - 5** | Average minus 5 | ~60-70% | Conservative lines (good safety margin) |
| **Avg + 5** | Average plus 5 | ~30-40% | Aggressive lines (higher odds) |
| **Avg + 10** | Average plus 10 | ~10-20% | Very aggressive lines (longshot bets) |

**Note:** The "Avg" column (exactly at average) has been removed for simplicity. Focus is on conservative (Avg-5, Avg-10) and aggressive (Avg+5, Avg+10) lines.

---

## 📋 Output Format

### 1. Top 30 Scorers - Points Hit Rates

```
==================================================================================================
TOP 30 SCORERS - POINTS HIT RATES
==================================================================================================

┌──────────────────────┬──────────────────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Player               │ Team             │ Avg PPG  │ Games  │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├──────────────────────┼──────────────────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│ LaMelo Ball          │ Charlotte Hornets│  31.1    │   16   │  93.8   │  75.0   │  31.3   │   12.5   │
│ Giannis Antetokounmpo│ Milwaukee Bucks  │  30.8    │   18   │  88.9   │  72.2   │  27.8   │   11.1   │
│ Anthony Davis        │ LA Lakers        │  29.2    │   17   │  88.2   │  70.6   │  29.4   │   11.8   │
│ ...                  │ ...              │  ...     │   ...  │  ...    │  ...    │  ...    │   ...    │
└──────────────────────┴──────────────────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘
```

### 2. Top 30 Assist Leaders - Assists Hit Rates

```
==================================================================================================
TOP 30 ASSIST LEADERS - ASSISTS HIT RATES
==================================================================================================

┌──────────────────────┬──────────────────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Player               │ Team             │ Avg APG  │ Games  │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├──────────────────────┼──────────────────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│ Trae Young           │ Atlanta Hawks    │  12.1    │   18   │  83.3   │  66.7   │  33.3   │   11.1   │
│ Nikola Jokić         │ Denver Nuggets   │  10.8    │   17   │  88.2   │  70.6   │  35.3   │   17.6   │
│ ...                  │ ...              │  ...     │   ...  │  ...    │  ...    │  ...    │   ...    │
└──────────────────────┴──────────────────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘
```

### 3. Top 30 Rebounders - Rebounds Hit Rates

```
==================================================================================================
TOP 30 REBOUNDERS - REBOUNDS HIT RATES
==================================================================================================

┌──────────────────────┬──────────────────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Player               │ Team             │ Avg RPG  │ Games  │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├──────────────────────┼──────────────────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│ Nikola Jokić         │ Denver Nuggets   │  13.7    │   17   │  94.1   │  76.5   │  29.4   │   11.8   │
│ Domantas Sabonis     │ Sacramento Kings │  13.5    │   18   │  88.9   │  72.2   │  27.8   │   11.1   │
│ ...                  │ ...              │  ...     │   ...  │  ...    │  ...    │  ...    │   ...    │
└──────────────────────┴──────────────────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘
```

**Reading the tables:**
- **Player**: Full name
- **Team**: Current team
- **Avg PPG/APG/RPG**: Season average for the stat
- **Games**: Number of games played (minimum 10)
- **Avg-10**: % of games going over (Average - 10)
- **Avg-5**: % of games going over (Average - 5)
- **Avg+5**: % of games going over (Average + 5)
- **Avg+10**: % of games going over (Average + 10)

### 4. League-Wide Summary

At the end, you get league-wide averages across all qualified players:

```
==================================================================================================
LEAGUE-WIDE HIT RATE SUMMARY
==================================================================================================

┌──────────────────┬─────────┬─────────┬─────────┬──────────┐
│ Stat             │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├──────────────────┼─────────┼─────────┼─────────┼──────────┤
│ Points           │  85.2   │  67.3   │  32.1   │   14.7   │
│ Assists          │  83.6   │  65.9   │  30.4   │   13.9   │
│ Rebounds         │  84.8   │  66.7   │  33.5   │   15.2   │
└──────────────────┴─────────┴─────────┴─────────┴──────────┘
```

---

## 💡 How to Use This Analysis

### 1. **Identify Conservative Opportunities**

Look for players with **high hit rates at Avg-5**:
- Hit rate ≥ 70% at Avg-5 = Very consistent performer
- Good for accumulator bets where you need reliability
- Lower odds but higher probability

**Example:**
```
Player: Giannis Antetokounmpo
Avg-5: 78.9%  ← Very reliable at this line
```

### 2. **Find Value Bets**

Look for players with **better than expected hit rates**:
- If Avg hit rate > 55%, player outperforms their average more often
- Suggests positive skew in performance distribution
- Line might be undervalued

**Example:**
```
Player: Anthony Edwards
Avg: 58.3%     ← Beats average 58% of time (expected ~50%)
Avg+5: 39.1%   ← Beats aggressive line 39% (expected ~30%)
```

### 3. **Avoid Inconsistent Players**

Look for players with **low hit rates at conservative lines**:
- Hit rate < 60% at Avg-5 = Inconsistent
- Risky for prop bets
- High variance

**Example:**
```
Player: Jordan Poole
Avg-5: 54.2%   ← Below expected ~60-70%
Avg: 42.1%     ← Below expected ~50%
```

### 4. **Line Shopping Strategy**

**Scenario:** Sportsbook offers "LeBron James Over 23.5 Points"
- LeBron's average: 25.8 PPG
- Line offset: 23.5 - 25.8 = -2.3 (roughly Avg-3)
- Expected hit rate: ~65% (between Avg-5 and Avg)
- Check table: If his Avg-5 shows 72% → Good bet
- If Avg-5 shows only 58% → Avoid, too inconsistent

### 5. **Build Accumulators**

Combine multiple players with:
- High hit rates at conservative lines (Avg-5 or Avg-10)
- Players averaging 70%+ at chosen line position
- Independent games (different matches)

**Example 3-Leg Acca:**
```
Leg 1: Player A - Avg-5 line, 75% hit rate
Leg 2: Player B - Avg-5 line, 72% hit rate
Leg 3: Player C - Avg-5 line, 78% hit rate

Expected probability: 0.75 × 0.72 × 0.78 = 42.1%
If odds > 2.38x (1/0.421), it's +EV!
```

---

## 🔍 Interpreting Results

### Normal Distribution Expectations

For normally distributed stats, expected hit rates should be:

| Line Position | Expected % Over | Standard Deviations |
|---------------|-----------------|---------------------|
| Avg - 10 | ~84% | -1.0σ |
| Avg - 5 | ~69% | -0.5σ |
| Avg | ~50% | 0.0σ |
| Avg + 5 | ~31% | +0.5σ |
| Avg + 10 | ~16% | +1.0σ |

### Red Flags

**If you see:**
- Avg hit rate < 45% → Player underperforming average (negative skew)
- Avg hit rate > 55% → Player outperforming average (positive skew)
- Avg-5 < 60% → High variance, inconsistent
- Avg+10 > 25% → Very positive skew, explosive potential

---

## 🎮 Real-World Example

### Case Study: LaMelo Ball

```
LaMelo Ball - Charlotte Hornets
Season Average: 31.1 PPG (16 games)

Hit Rates:
- Avg-10 (21.1 PTS): 93.8%  ← Almost always beats this
- Avg-5  (26.1 PTS): 75.0%  ← Very reliable
- Avg    (31.1 PTS): 56.3%  ← Slightly outperforms
- Avg+5  (36.1 PTS): 31.3%  ← On target
- Avg+10 (41.1 PTS): 12.5%  ← Rare explosions
```

**Analysis:**
1. **Consistency**: 75% at Avg-5 → Very consistent scorer
2. **Positive skew**: 56.3% at Avg → Beats average more than expected
3. **Upside**: 12.5% at Avg+10 → Has explosive games
4. **Betting strategy**:
   - Conservative: Use 26.5 line (Avg-5), expect ~75% hit rate
   - Balanced: Use 31.5 line (Avg), expect ~55% hit rate
   - Aggressive: Use 36.5 line (Avg+5), expect ~30% hit rate

**If sportsbook offers "LaMelo Ball Over 28.5 PTS":**
- Offset: 28.5 - 31.1 = -2.6 (close to Avg-3)
- Expected hit rate: ~65% (interpolate between Avg-5 and Avg)
- If odds are 1.50x (67% implied probability) → **SKIP** (bad value)
- If odds are 1.60x (62.5% implied probability) → **BET** (positive EV)

---

## 📁 Where to Find Results

### In Console Output

The analysis runs automatically as **Step 9a** in the main analysis pipeline:
```
====================================================================================================
LINE HIT RATE ANALYSIS BY TEAM
====================================================================================================
```

### In Returned Data

The results are stored in `line_hit_rates` dictionary:
```python
# After running the analysis
line_hit_rates  # Dictionary with team names as keys

# Access specific team
hawks_data = line_hit_rates['Atlanta Hawks']

# View all columns
print(hawks_data.columns)

# Filter for specific player
trae_data = hawks_data[hawks_data['player_name'].str.contains('Trae Young')]
```

---

## 🔧 Customization

### Change Minimum Games Threshold

Edit line 985 in `dataedits.py`:
```python
# Current: ≥10 games
qualified_players = prop_with_team[prop_with_team['games_played'] >= 10].copy()

# Change to ≥20 games for more reliable data
qualified_players = prop_with_team[prop_with_team['games_played'] >= 20].copy()
```

### Change Line Offsets

Edit lines 967-973 in `dataedits.py`:
```python
# Current offsets
offsets = {
    'Avg-10': -10,
    'Avg-5': -5,
    'Avg': 0,
    'Avg+5': +5,
    'Avg+10': +10
}

# Add more granular offsets
offsets = {
    'Avg-15': -15,
    'Avg-10': -10,
    'Avg-5': -5,
    'Avg-2': -2,
    'Avg': 0,
    'Avg+2': +2,
    'Avg+5': +5,
    'Avg+10': +10,
    'Avg+15': +15
}
```

### Add More Stats

Edit lines 959-964 in `dataedits.py`:
```python
# Add blocks and steals
stats_to_analyze = {
    'PTS': {'col': 'PTS', 'avg_col': 'PPG', 'name': 'Points'},
    'AST': {'col': 'AST', 'avg_col': 'APG', 'name': 'Assists'},
    'REB': {'col': 'REB', 'avg_col': 'RPG', 'name': 'Rebounds'},
    'FG3M': {'col': 'FG3M', 'avg_col': 'FG3M_PG', 'name': '3-Pointers Made'},
    'BLK': {'col': 'BLK', 'avg_col': 'BPG', 'name': 'Blocks'},
    'STL': {'col': 'STL', 'avg_col': 'SPG', 'name': 'Steals'}
}
```

---

## ✅ Summary

**This analysis helps you:**
1. ✅ Identify reliable players for conservative lines
2. ✅ Find value bets where players outperform expectations
3. ✅ Avoid inconsistent players with high variance
4. ✅ Determine optimal line positions for props
5. ✅ Build data-driven accumulator bets
6. ✅ Understand player performance distributions

**Key metric: Hit Rate**
> **% of games player goes OVER the specified line**

**Critical for:**
- Line shopping
- Accumulator building
- Value identification
- Risk management
- Expected value calculations
