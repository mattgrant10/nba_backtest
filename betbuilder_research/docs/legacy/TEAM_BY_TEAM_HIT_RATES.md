# Team-by-Team Hit Rate Analysis

## Overview

The **Line Hit Rate Analysis** now displays hit rates **organized by team** with **3 separate tables per team**:

1. **Points Hit Rates** - Top 15 scorers on the team
2. **Assists Hit Rates** - Top 15 playmakers on the team
3. **Rebounds Hit Rates** - Top 15 rebounders on the team

Each table shows hit rates at **+/-5 and +/-10** from the player's season average.

---

## 🎯 What is Hit Rate?

**Hit Rate Definition:**
> The percentage of games where a player's actual result was **OVER** the specified line.

**Example:**
- Player averages 20.0 PPG over 20 games
- In 15 games, they scored more than 25 points (Avg+5)
- **Hit Rate at Avg+5 = 75%** (15/20 games)

---

## 📊 Output Format

### For Each Team

You'll see **3 tables** displayed sequentially:

```
==================================================================================================
Atlanta Hawks
==================================================================================================

>>> POINTS HIT RATES (% of games OVER line)

┌───────────────────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Player            │ Avg PPG  │ Games  │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├───────────────────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│ Trae Young        │  28.5    │   18   │  88.9   │  72.2   │  33.3   │   16.7   │
│ Jalen Johnson     │  18.9    │   18   │  94.4   │  72.2   │  27.8   │   11.1   │
│ De'Andre Hunter   │  16.2    │   17   │  82.4   │  64.7   │  29.4   │   17.6   │
│ ...               │  ...     │   ...  │  ...    │  ...    │  ...    │   ...    │
└───────────────────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘

>>> ASSISTS HIT RATES (% of games OVER line)

┌───────────────────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Player            │ Avg APG  │ Games  │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├───────────────────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│ Trae Young        │  12.1    │   18   │  83.3   │  66.7   │  33.3   │   11.1   │
│ Jalen Johnson     │   5.6    │   18   │  77.8   │  61.1   │  27.8   │   11.1   │
│ Dyson Daniels     │   3.8    │   16   │  81.3   │  62.5   │  31.3   │   12.5   │
│ ...               │  ...     │   ...  │  ...    │  ...    │  ...    │   ...    │
└───────────────────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘

>>> REBOUNDS HIT RATES (% of games OVER line)

┌───────────────────┬──────────┬────────┬─────────┬─────────┬─────────┬──────────┐
│ Player            │ Avg RPG  │ Games  │ Avg-10  │ Avg-5   │ Avg+5   │ Avg+10   │
├───────────────────┼──────────┼────────┼─────────┼─────────┼─────────┼──────────┤
│ Jalen Johnson     │  10.1    │   18   │  88.9   │  72.2   │  27.8   │   11.1   │
│ Clint Capela      │   9.8    │   17   │  88.2   │  70.6   │  29.4   │   11.8   │
│ Onyeka Okongwu    │   7.2    │   15   │  86.7   │  66.7   │  33.3   │   13.3   │
│ ...               │  ...     │   ...  │  ...    │  ...    │  ...    │   ...    │
└───────────────────┴──────────┴────────┴─────────┴─────────┴─────────┴──────────┘
```

This pattern repeats for all 30 teams.

---

## 📋 Column Definitions

| Column | Description | Example |
|--------|-------------|---------|
| **Player** | Full player name | Trae Young |
| **Avg PPG/APG/RPG** | Season average for the stat | 28.5 PPG |
| **Games** | Games played (minimum 10) | 18 |
| **Avg-10** | % of games going over (Average - 10) | 88.9% |
| **Avg-5** | % of games going over (Average - 5) | 72.2% |
| **Avg+5** | % of games going over (Average + 5) | 33.3% |
| **Avg+10** | % of games going over (Average + 10) | 16.7% |

---

## 📏 Understanding the Hit Rates

### **Avg-10 (Very Conservative)**
- Line is 10 points/assists/rebounds **below** season average
- Expected hit rate: ~80-90%
- Use case: High-confidence plays, accumulator foundations

**Example:** Player averages 25 PPG, line is 15 points
- Should beat this line in ~85% of games

### **Avg-5 (Conservative)**
- Line is 5 points/assists/rebounds **below** season average
- Expected hit rate: ~60-70%
- Use case: Safe bets, reliable props

**Example:** Player averages 25 PPG, line is 20 points
- Should beat this line in ~65% of games

### **Avg+5 (Aggressive)**
- Line is 5 points/assists/rebounds **above** season average
- Expected hit rate: ~30-40%
- Use case: Higher odds, riskier plays

**Example:** Player averages 25 PPG, line is 30 points
- Should beat this line in ~35% of games

### **Avg+10 (Very Aggressive)**
- Line is 10 points/assists/rebounds **above** season average
- Expected hit rate: ~10-20%
- Use case: Longshot bets, explosion plays

**Example:** Player averages 25 PPG, line is 35 points
- Should beat this line in ~15% of games

---

## 💡 How to Use (Team-by-Team)

### **1. Find Reliable Team Players for Accumulators**

**Goal:** Build a 3-leg accumulator using different teams

**Look for:**
- High hit rates at **Avg-5** (70%+ ideal)
- Players from different teams (different games)
- Consistent across multiple games (games column)

**Example:**
```
Team 1 - Hawks: Trae Young PTS Avg-5 = 72.2%
Team 2 - Lakers: LeBron James PTS Avg-5 = 75.0%
Team 3 - Celtics: Jayson Tatum PTS Avg-5 = 68.9%

Combined probability: 0.722 × 0.750 × 0.689 = 37.3%
If odds > 2.68x → Positive EV bet!
```

### **2. Compare Players Within a Team**

**Goal:** Decide which teammate to bet on

**Look at:** Same team, same stat type

**Example - Hawks Points:**
```
Trae Young:     Avg-5 = 72.2%, Avg+5 = 33.3%
Jalen Johnson:  Avg-5 = 72.2%, Avg+5 = 27.8%
De'Andre Hunter: Avg-5 = 64.7%, Avg+5 = 29.4%
```

**Analysis:**
- Trae and Jalen equally consistent at Avg-5
- Trae has more upside (33.3% at Avg+5 vs 27.8%)
- De'Andre less consistent overall
- **Decision:** Pick Trae for balanced risk/reward

### **3. Identify Team Strengths**

**Goal:** Find which stats a team excels at

**Look at:** Compare hit rates across all 3 tables for one team

**Example - Team Analysis:**
```
Hawks Points:   Multiple players with 70%+ at Avg-5
Hawks Assists:  Trae at 66.7% but others lower
Hawks Rebounds: Decent at 70%+ for top 2 players

Conclusion: Hawks are a scoring team, strong on rebounds,
            but assist props risky outside of Trae
```

### **4. Spot Value Bets**

**Goal:** Find players beating expectations

**Look for:**
- Avg+5 hit rate > 40% (outperforming average)
- Avg-5 hit rate > 75% (very reliable)
- Avg+10 hit rate > 25% (explosive potential)

**Example:**
```
Player: Giannis Antetokounmpo
Avg+5: 45.0%  ← Way above expected ~35%
Avg+10: 28.0% ← Above expected ~15%

Analysis: Player frequently exceeds their average
          Positive skew in performance
          Good for aggressive lines
```

### **5. Build Same-Game Parlays (SGP)**

**Goal:** Combine multiple props from the same team

**Look at:** All 3 tables for one team

**Example - Lakers SGP:**
```
LeBron James:  PTS Avg-5 = 75.0%
Anthony Davis: REB Avg-5 = 72.0%
D'Angelo Russell: AST Avg-5 = 68.0%

All from same game, different stats
Combined: 0.75 × 0.72 × 0.68 = 36.7%
```

---

## 🎯 Real-World Example

### **Case Study: Building a 5-Leg Accumulator**

**Goal:** Find 5 players from 5 different teams with 70%+ at Avg-5

**Step 1: Scan multiple teams**
```
Atlanta Hawks - POINTS table:
Trae Young: 72.2% at Avg-5 ✓ (Select: O23.5 PTS)

Boston Celtics - POINTS table:
Jayson Tatum: 75.0% at Avg-5 ✓ (Select: O25.5 PTS)

LA Lakers - REBOUNDS table:
Anthony Davis: 72.0% at Avg-5 ✓ (Select: O7.5 REB)

Milwaukee Bucks - POINTS table:
Giannis: 78.0% at Avg-5 ✓ (Select: O25.5 PTS)

Denver Nuggets - ASSISTS table:
Nikola Jokić: 71.0% at Avg-5 ✓ (Select: O5.5 AST)
```

**Step 2: Calculate expected probability**
```
0.722 × 0.750 × 0.720 × 0.780 × 0.710 = 21.8%
```

**Step 3: Compare to odds**
```
If betting odds are 5.50x (implied prob = 18.2%)
→ Your edge: 21.8% - 18.2% = +3.6% positive EV!
```

---

## 📊 League-Wide Summary

At the end of all team tables, you'll see:

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

**Use this to:**
- Compare individual players to league average
- Identify outliers (players way above/below average)
- Validate your understanding of hit rate expectations

---

## 🚀 How to Run

```bash
python data/dataedits.py
```

The analysis runs automatically as **Step 9a: Line Hit Rate Analysis**

**Output:**
- 30 teams
- 3 tables per team (Points, Assists, Rebounds)
- Top 15 players per table (sorted by average)
- = **90 tables total** covering all teams and all major stats

---

## 📁 Accessing Results

After running, the results are stored in the `line_hit_rates` dictionary:

```python
# Access all data
line_hit_rates['all_players']  # DataFrame with all players

# Access specific team
hawks_data = line_hit_rates['by_team']['Atlanta Hawks']

# Filter for specific player
trae = hawks_data[hawks_data['player_name'].str.contains('Trae')]

# View points hit rates
print(trae[['player_name', 'PTS_avg', 'PTS_Avg-5', 'PTS_Avg+5']])

# Get league-wide summary
print(line_hit_rates['summary'])
```

---

## ✅ Summary

**Format:**
- **Team-by-team** organization (30 teams)
- **3 tables per team** (Points, Assists, Rebounds)
- **Top 15 players** per table
- **4 hit rate columns** (Avg-10, Avg-5, Avg+5, Avg+10)

**Benefits:**
- Easy to compare players within a team
- Quick team strength identification
- Perfect for same-game parlays
- Organized for team-specific betting

**Use Cases:**
- Build accumulators from different teams
- Compare teammates
- Identify team strengths/weaknesses
- Find value in specific game contexts
