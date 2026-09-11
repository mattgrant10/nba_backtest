# Copula Plots Guide

## Overview

The copula analysis now includes **fitted copula visualizations** showing different copula families. Copulas are statistical tools that model the dependence structure between variables independently of their marginal distributions.

---

## New Copula Plots Added

### 1. **Fitted Gaussian Copula Plots** (`copula_fitted_gaussian.png`)

**What it shows:**
- Top 4 variable pairs by Spearman correlation
- Empirical copula data (scatter of rank-transformed data)
- Gaussian copula density contours overlaid on the empirical data
- Tail dependency annotations

**How to read:**
- **Blue scatter points**: Actual empirical copula (rank-transformed data)
- **Red contour lines**: Fitted Gaussian copula density
- **Black dashed line**: Independence reference (if data falls on this line, variables are independent)
- **Contour labels**: Density values (higher = stronger dependency)

**Use case:**
- Verify if the Gaussian copula is a good fit for the data
- Compare theoretical (Gaussian) vs empirical copula structure
- Identify linear vs non-linear dependencies

---

### 2. **Copula Family Comparison** (`copula_family_comparison.png`)

**What it shows:**
A 2×2 grid comparing different copula families for the **top variable pair**:

1. **Empirical Copula (KDE)**
   - Kernel density estimation of the actual data
   - Shows the true dependence structure

2. **Gaussian Copula**
   - Symmetric dependence
   - No tail dependence
   - Good for normally distributed data

3. **Clayton Copula**
   - **Lower tail dependence** (stronger correlation in bottom 10%)
   - Good when variables move together during poor performance
   - Parameter θ shown (higher = stronger lower tail dependence)

4. **Gumbel Copula**
   - **Upper tail dependence** (stronger correlation in top 10%)
   - Good when variables move together during strong performance
   - Parameter θ shown (higher = stronger upper tail dependence)

**How to read:**
- Compare the empirical copula (top-left) to each fitted copula
- Look for which copula family best matches the data pattern
- Check tail behavior (corners of the plots)

---

## Copula Families Explained

### **Gaussian Copula**
- **Properties**: Symmetric, no tail dependence
- **Best for**: Linear relationships, normally distributed data
- **Limitations**: Underestimates extreme events (tail risks)

**Formula used:**
```
C(u,v) = Φ_ρ(Φ^(-1)(u), Φ^(-1)(v))
```
where Φ is the standard normal CDF and ρ is the correlation parameter.

---

### **Clayton Copula**
- **Properties**: Lower tail dependence, asymmetric
- **Best for**: Variables that crash together (e.g., points and assists both low)
- **Applications**: Risk management, downside correlation

**Characteristic:**
- Higher density in the **lower-left corner**
- When one variable is low, the other tends to be low too

**Formula used:**
```
C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
```
where θ > 0 is the dependence parameter.

---

### **Gumbel Copula**
- **Properties**: Upper tail dependence, asymmetric
- **Best for**: Variables that peak together (e.g., points and assists both high)
- **Applications**: Extreme value analysis, upside correlation

**Characteristic:**
- Higher density in the **upper-right corner**
- When one variable is high, the other tends to be high too

**Formula used:**
```
C(u,v) = exp(-((-ln u)^θ + (-ln v)^θ)^(1/θ))
```
where θ ≥ 1 is the dependence parameter.

---

## Integration with Existing Copula Plots

The new fitted copula plots complement the existing copula visualizations:

### **Existing Plots:**
1. `copula_dependencies.png` - Raw scatter plots of top 6 pairs
2. `copula_rank_transformed.png` - Hexbin density of rank-transformed data
3. `copula_heatmap.png` - Spearman correlation matrix
4. `copula_tail_dependencies.png` - Upper vs lower tail correlation comparison

### **New Plots:**
5. `copula_fitted_gaussian.png` - Gaussian copula fits for top 4 pairs
6. `copula_family_comparison.png` - 4 copula families compared for top pair

**Total copula plots: 6**

---

## Practical Applications for NBA Betting

### 1. **Identify Joint Behavior**
- If PTS and AST have **Gumbel copula** fit → They peak together
  - Strategy: Bet on both going over when player has strong matchup
- If PTS and AST have **Clayton copula** fit → They crash together
  - Strategy: Avoid betting on one when other is expected to be low

### 2. **Multi-Leg Accumulator Risk**
- **Gaussian copula** between props → Standard correlation applies
- **Clayton copula** → Higher risk of all props failing together
- **Gumbel copula** → Higher chance of all props hitting together

### 3. **Tail Risk Assessment**
- Check lower tail dependence for defensive betting (avoiding total collapse)
- Check upper tail dependence for aggressive betting (all props hit together)

---

## How to Interpret the Plots

### **Good Gaussian Copula Fit:**
✓ Empirical data scatter matches the Gaussian contour pattern
✓ Data evenly distributed around contours
✓ No strong concentration in corners

### **Poor Gaussian Copula Fit (consider other families):**
✗ Data concentrated in lower-left corner → Use **Clayton copula**
✗ Data concentrated in upper-right corner → Use **Gumbel copula**
✗ Data shows non-symmetric pattern → Use asymmetric copula

### **Example Interpretation:**

**Scenario:** PTS vs AST plot shows:
- Empirical data heavily concentrated in upper-right
- Gumbel copula contours match better than Gaussian
- Upper tail dependency = 0.85 (very high)

**Interpretation:**
- When a player has a high-scoring game (top 10%), they are very likely to also have high assists (85% correlation in top 10%)
- This is stronger than the overall correlation suggests
- **Betting strategy**: When conditions favor a big scoring night, also bet on over for assists

---

## Technical Notes

### **Rank Transformation:**
All copula plots use rank-transformed data:
```python
x_rank = rankdata(x) / (len(x) + 1)  # Transforms to [0, 1]
y_rank = rankdata(y) / (len(y) + 1)
```

This removes the effect of marginal distributions and shows only the dependence structure.

### **Parameter Estimation:**
- **Gaussian**: Uses Spearman's ρ directly
- **Clayton**: θ ≈ 2ρ / (1 - ρ)
- **Gumbel**: θ ≈ 1 / (1 - ρ)

These are approximations based on Spearman's rank correlation.

### **Density Contours:**
- Higher density (inner contours) = more data points
- Lower density (outer contours) = fewer data points
- Contour shape reveals dependence pattern

---

## Where to Find the Plots

**File locations:**
```
output/figures/copula_fitted_gaussian.png
output/figures/copula_family_comparison.png
```

**In the PDF report:**
- Located in the "Copula Analysis" section
- Comes after the rank-transformed plots and heatmaps
- Before the team consistency plots

---

## Summary

**What changed:**
✅ Added Gaussian copula fitting for top 4 variable pairs
✅ Added copula family comparison (Empirical, Gaussian, Clayton, Gumbel)
✅ Visualized theoretical copula densities vs empirical data
✅ Integrated into main PDF report automatically

**Benefits:**
- Better understanding of dependence structure
- Identify tail dependencies for risk assessment
- Choose appropriate copula family for simulation/modeling
- Improved betting strategy for correlated props

**Total copula visualizations: 6 figures**
- 3 existing (dependencies, rank-transformed, heatmap, tail comparison)
- 2 new (fitted Gaussian, family comparison)
