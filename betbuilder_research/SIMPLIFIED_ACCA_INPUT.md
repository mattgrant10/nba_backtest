# Simplified Accumulator Input Format

## Summary

The accumulator input format has been **dramatically simplified**. You now only need to provide the essential information, and the system automatically looks up results from the dataframe.

---

## ❌ OLD FORMAT (Complex - Too Much Info Required)

```python
{
    'leg': 2,                                    # Manual numbering
    'date': '2024-11-22',
    'time': '22:00',                            # Not used
    'match': 'New York Knicks @ Orlando Magic', # Only for display
    'odds': 4.90,
    'props': [
        {
            'player': 'Wendell Carter',
            'stat': 'PTS',
            'line': 7.5,
            'over': True,
            'result': 11                         # Had to manually enter
        },
        {
            'player': 'Mikal Bridges',
            'stat': 'PTS',
            'line': 11.5,
            'over': True,
            'result': 18                         # Had to manually enter
        },
        # ... more props
    ]
}
```

**Problems:**
- ❌ Required manual entry of 'result' for each prop
- ❌ Had to provide 'leg' number manually
- ❌ Had to provide 'match' name (only used for display)
- ❌ Had to provide 'time' (not even used!)
- ❌ Error-prone: Easy to mistype results

---

## ✅ NEW FORMAT (Simple - Automatic Lookup)

```python
{
    'date': '2024-11-22',
    'odds': 4.90,  # Optional - defaults to 1.0 if not provided
    'props': [
        {
            'player': 'Wendell Carter',
            'stat': 'PTS',
            'line': 7.5,
            'over': True
        },
        {
            'player': 'Mikal Bridges',
            'stat': 'PTS',
            'line': 11.5,
            'over': True
        },
        # ... more props
    ]
}
```

**Benefits:**
- ✅ **Results looked up automatically** from dataframe
- ✅ **Leg numbers auto-generated** (1, 2, 3, ...)
- ✅ **No match name needed** (displayed as date instead)
- ✅ **No time needed** (never used anyway)
- ✅ **Less typing, fewer errors**

---

## 📋 Required vs Optional Fields

### **REQUIRED Fields:**

```python
{
    'date': 'YYYY-MM-DD',  # Date of the game
    'props': [              # List of player props
        {
            'player': 'Player Name',  # Full or partial name (last name works)
            'stat': 'PTS',            # PTS, AST, REB, BLK, STL, FG3M, TOV, MIN
            'line': 7.5,              # Prop line value
            'over': True              # True for Over, False for Under
        }
    ]
}
```

### **OPTIONAL Fields:**

```python
{
    'odds': 4.90  # Betting odds - defaults to 1.0 if not provided
}
```

### **REMOVED Fields (No Longer Needed):**

- ❌ `'leg'` - Auto-generated (1, 2, 3, ...)
- ❌ `'time'` - Never used
- ❌ `'match'` - Date is sufficient
- ❌ `'result'` inside props - **Automatically looked up from dataframe!**

---

## 🎯 Supported Stats

You can use any of these stat types:

| Stat Code | Description |
|-----------|-------------|
| `'PTS'` | Points |
| `'AST'` | Assists |
| `'REB'` | Rebounds |
| `'BLK'` | Blocks |
| `'STL'` | Steals |
| `'FG3M'` | Three-Pointers Made |
| `'TOV'` | Turnovers |
| `'MIN'` | Minutes Played |

---

## 📝 Complete Example

### Before (8 fields per leg + result for each prop):

```python
historical_accas = [
    {
        'leg': 1,
        'date': '2024-11-22',
        'time': '18:00',
        'match': 'Los Angeles Clippers @ Charlotte Hornets',
        'odds': 2.70,
        'props': [
            {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 12.5, 'over': True, 'result': 18},
            {'player': 'Miles Bridges', 'stat': 'PTS', 'line': 11.5, 'over': True, 'result': 19},
        ]
    },
    {
        'leg': 2,
        'date': '2024-11-22',
        'time': '22:00',
        'match': 'New York Knicks @ Orlando Magic',
        'odds': 4.90,
        'props': [
            {'player': 'Franz Wagner', 'stat': 'PTS', 'line': 17.5, 'over': True, 'result': 37},
            {'player': 'Jalen Brunson', 'stat': 'PTS', 'line': 21.5, 'over': True, 'result': 33},
        ]
    }
]
```

### After (2-3 fields per leg, no results needed):

```python
historical_accas = [
    {
        'date': '2024-11-22',
        'odds': 2.70,  # Optional
        'props': [
            {'player': 'LaMelo Ball', 'stat': 'PTS', 'line': 12.5, 'over': True},
            {'player': 'Miles Bridges', 'stat': 'PTS', 'line': 11.5, 'over': True},
        ]
    },
    {
        'date': '2024-11-22',
        'odds': 4.90,  # Optional
        'props': [
            {'player': 'Franz Wagner', 'stat': 'PTS', 'line': 17.5, 'over': True},
            {'player': 'Jalen Brunson', 'stat': 'PTS', 'line': 21.5, 'over': True},
        ]
    }
]
```

**Reduction: ~60% less typing, zero manual result entry!**

---

## 🔧 How It Works

When you run the analysis, the function:

1. **Converts the date** to datetime format
2. **Finds all games** played on that date
3. **Searches for each player** by name (supports partial matching)
4. **Looks up the actual result** from the dataframe automatically
5. **Evaluates whether the prop hit** (over/under)
6. **Calculates margins** and consistency metrics
7. **Displays detailed logging** with actual player names and results

---

## 🚨 Error Handling

The function handles common issues gracefully:

### **No games found on date:**
```
⚠️  No games found on 2024-11-23, skipping leg 2
```

### **Player didn't play:**
```
⚠️  LaMelo Ball did not play on 2024-11-22, skipping prop
```

### **Unsupported stat type:**
```
⚠️  Unsupported stat type: FANTASY_PTS, skipping
```

### **No props evaluated:**
```
⚠️  No props could be evaluated for leg 3, skipping
```

---

## 📊 Output Format

The analysis provides:

### **Leg-by-leg logging:**
```
>>> Analyzing Leg 1: 2024-11-22
    Games that night: 8
    LaMelo Ball PTS O12.5: 18.0 - ✓ (margin: +5.5)
    Miles Bridges PTS O11.5: 19.0 - ✓ (margin: +7.5)
    ✓✓✓ LEG WON (2/2 props hit = 100.0%)
```

### **Summary table:**
```
┌──────┬────────┬───────┬────────┬───────┬────────┬───────────────┐
│ Leg  │ Props  │ Hit   │ Hit%   │ Won   │ Odds   │ Implied Prob  │
├──────┼────────┼───────┼────────┼───────┼────────┼───────────────┤
│ 1    │ 2      │ 2     │ 100.0  │ True  │ 2.70   │ 37.0          │
│ 2    │ 7      │ 6     │ 85.7   │ False │ 4.90   │ 20.4          │
│ 3    │ 7      │ 7     │ 100.0  │ True  │ 3.80   │ 26.3          │
└──────┴────────┴───────┴────────┴───────┴────────┴───────────────┘
```

### **Individual prop details:**
```
┌──────┬──────────────────┬────────┬────────┬──────────┬──────────┬───────┬────────┬───────┐
│ Leg  │ Player           │ Stat   │ Line   │ Result   │ Margin   │ Hit   │ Avg    │ CV    │
├──────┼──────────────────┼────────┼────────┼──────────┼──────────┼───────┼────────┼───────┤
│ 1    │ LaMelo Ball      │ PTS    │ 12.5   │ 18.0     │ +5.5     │ True  │ 31.1   │ 0.23  │
│ 1    │ Miles Bridges    │ PTS    │ 11.5   │ 19.0     │ +7.5     │ True  │ 19.8   │ 0.31  │
└──────┴──────────────────┴────────┴────────┴──────────┴──────────┴───────┴────────┴───────┘
```

---

## 💡 Tips

1. **Use last names only** - `'Ball'` works just as well as `'LaMelo Ball'`
2. **Odds are optional** - If you're just testing prop performance without caring about betting odds, omit them
3. **Multiple props per player** - You can have the same player multiple times with different stats
4. **Date format matters** - Always use `'YYYY-MM-DD'` format (e.g., `'2024-11-22'`)
5. **Case insensitive** - Player names are matched case-insensitively

---

## 🎯 Where to Edit

The input is at the top of the `analyze_historical_accas()` function:

**File:** `data/dataedits.py`
**Function:** `analyze_historical_accas()`
**Lines:** ~2327-2363

Look for this section:
```python
# ============================================================================
# 🎯 SIMPLIFIED ACCUMULATOR INPUT - EDIT BELOW
# ============================================================================

historical_accas = [
    # Your accumulator legs here
]
```

---

## ✅ Summary

**What changed:**
- ✅ Removed 'result' from props (auto-lookup)
- ✅ Removed 'leg' from legs (auto-generated)
- ✅ Removed 'time' from legs (unused)
- ✅ Removed 'match' from legs (replaced with date)
- ✅ Made 'odds' optional (defaults to 1.0)

**What you need now:**
- Just `date` and `props` with `player`, `stat`, `line`, `over`
- Optionally provide `odds` if you want implied probability analysis

**Result:**
- ~60% less typing
- Zero manual result entry
- Automatic lookup from dataframe
- Less error-prone
- Cleaner, simpler code
