"""Preserved research calculations: dependence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tabulate import tabulate
from src.config import get_logger

logger = get_logger(__name__)


def analyze_advanced_dependencies(df: pd.DataFrame) -> tuple:
    """
    Advanced dependency analysis using entropy, mutual information, and copulas.

    Returns:
        - entropy_results: DataFrame with entropy measures
        - mi_matrix: Mutual information matrix
        - copula_results: DataFrame with copula dependency measures
    """
    from scipy.stats import entropy as scipy_entropy
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import spearmanr, kendalltau

    logger.info("\n" + "=" * 100)
    logger.info("ADVANCED DEPENDENCY ANALYSIS")
    logger.info("Entropy | Mutual Information | Copula Analysis")
    logger.info("=" * 100)

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove columns with too many missing values
    valid_cols = [col for col in numeric_cols if df[col].notna().sum() > 100]

    logger.info(f"\n>>> Analyzing {len(valid_cols)} numeric variables")

    # =========================================================================
    # STEP 1: ENTROPY ANALYSIS
    # =========================================================================
    logger.info("\n>>> STEP 1: ENTROPY ANALYSIS (Information Content)")
    logger.info("    Measuring uncertainty/randomness in each variable...")

    entropy_results = []

    for col in valid_cols:
        data = df[col].dropna()

        if len(data) < 2:
            continue

        # Discretize continuous data for entropy calculation
        # Use Freedman-Diaconis rule for bin width
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        n_bins = int(np.ceil((data.max() - data.min()) / bin_width)) if bin_width > 0 else 10
        n_bins = min(max(n_bins, 5), 50)  # Clamp between 5 and 50

        # Create histogram
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        hist = hist / hist.sum()  # Normalize to probability distribution

        # Calculate entropy (in bits)
        entr = scipy_entropy(hist, base=2)

        # Calculate normalized entropy (0 to 1)
        max_entropy = np.log2(n_bins)
        normalized_entropy = entr / max_entropy if max_entropy > 0 else 0

        # Calculate coefficient of variation
        cv = data.std() / data.mean() if data.mean() != 0 else np.inf

        logger.info(f"    {col}:")
        logger.info(f"        Entropy: {entr:.4f} bits (normalized: {normalized_entropy:.4f})")
        logger.info(f"        CV: {cv:.4f}")
        logger.info(
            f"        Interpretation: {'High variability' if normalized_entropy > 0.7 else 'Moderate variability' if normalized_entropy > 0.4 else 'Low variability'}"
        )

        entropy_results.append(
            {
                "Variable": col,
                "Entropy_Bits": entr,
                "Normalized_Entropy": normalized_entropy,
                "Coefficient_of_Variation": cv,
                "N_Bins": n_bins,
                "Variability": "High"
                if normalized_entropy > 0.7
                else "Moderate"
                if normalized_entropy > 0.4
                else "Low",
            }
        )

    entropy_df = pd.DataFrame(entropy_results)

    logger.info("\n    ENTROPY RESULTS TABLE:")
    print(tabulate(entropy_df.round(4), headers="keys", tablefmt="grid", showindex=False))

    # =========================================================================
    # STEP 2: MUTUAL INFORMATION ANALYSIS
    # =========================================================================
    logger.info("\n>>> STEP 2: MUTUAL INFORMATION ANALYSIS (Variable Dependencies)")
    logger.info("    Computing pairwise mutual information between variables...")

    # Prepare data for MI calculation
    df_valid = df[valid_cols].dropna()

    if len(df_valid) < 50:
        logger.warning("    Insufficient data for mutual information analysis")
        mi_matrix = pd.DataFrame()
    else:
        logger.info(f"    Using {len(df_valid):,} complete observations")

        # Compute mutual information matrix
        mi_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols, dtype=float)

        # Alias columns contain identical arrays. Cache only exact ordered inputs;
        # retain estimator direction, dtype, seed and complete-case population.
        column_keys = {
            c: (str(df_valid[c].dtype), df_valid[c].to_numpy().tobytes()) for c in valid_cols
        }
        mi_cache = {}
        for i, col1 in enumerate(valid_cols):
            for j, col2 in enumerate(valid_cols):
                if i == j:
                    # Self-MI equals entropy
                    mi_matrix.loc[col1, col2] = entropy_df[entropy_df["Variable"] == col1][
                        "Entropy_Bits"
                    ].values[0]
                elif i < j:
                    # Compute MI
                    X = df_valid[[col1]].values
                    y = df_valid[col2].values

                    try:
                        key = (column_keys[col1], column_keys[col2])
                        if key not in mi_cache:
                            mi_cache[key] = mutual_info_regression(X, y, random_state=42)[0]
                        mi_value = mi_cache[key]
                        mi_matrix.loc[col1, col2] = mi_value
                        mi_matrix.loc[col2, col1] = mi_value

                        if mi_value > 1.0:
                            logger.info(
                                f"    Strong dependency: {col1} ↔ {col2} (MI={mi_value:.4f})"
                            )
                    except (ValueError, TypeError) as error:
                        logger.warning("MI unavailable for %s/%s: %s", col1, col2, error)
                        mi_matrix.loc[col1, col2] = np.nan
                        mi_matrix.loc[col2, col1] = np.nan

        # Convert to numeric
        mi_matrix = mi_matrix.astype(float)

        # Display top dependencies
        logger.info("\n    TOP 10 MUTUAL INFORMATION PAIRS:")
        mi_pairs = []
        for i in range(len(valid_cols)):
            for j in range(i + 1, len(valid_cols)):
                val = mi_matrix.iloc[i, j]
                if not np.isnan(val):
                    mi_pairs.append({"Var1": valid_cols[i], "Var2": valid_cols[j], "MI_Score": val})

        mi_pairs_df = pd.DataFrame(mi_pairs).sort_values("MI_Score", ascending=False).head(10)
        print(tabulate(mi_pairs_df.round(4), headers="keys", tablefmt="grid", showindex=False))

    # =========================================================================
    # STEP 3: COPULA ANALYSIS (Tail Dependencies)
    # =========================================================================
    logger.info("\n>>> STEP 3: COPULA ANALYSIS (Non-linear Dependencies)")
    logger.info("    Computing rank correlations for copula-based dependency structure...")

    copula_results = []

    # Select key variables for copula analysis (avoid computation explosion)
    key_vars = ["PTS", "AST", "REB", "FG_PCT", "FG3_PCT", "MIN", "TOV", "STL", "BLK"]
    key_vars = [v for v in key_vars if v in valid_cols]

    logger.info(f"    Analyzing {len(key_vars)} key variables")

    df_copula = df[key_vars].dropna()

    if len(df_copula) < 50:
        logger.warning("    Insufficient data for copula analysis")
        copula_df = pd.DataFrame()
    else:
        logger.info(f"    Using {len(df_copula):,} complete observations")

        for i, var1 in enumerate(key_vars):
            for j, var2 in enumerate(key_vars):
                if i >= j:
                    continue

                x = df_copula[var1].values
                y = df_copula[var2].values

                # Compute rank correlations
                spearman_rho, spearman_p = spearmanr(x, y)
                kendall_tau, kendall_p = kendalltau(x, y)

                # Compute Pearson for comparison
                pearson_r = np.corrcoef(x, y)[0, 1]

                # Tail dependency index (simplified)
                # Upper tail: correlation in top 10%
                q90 = np.percentile(x, 90)
                upper_tail_mask = x >= q90
                if upper_tail_mask.sum() > 10:
                    upper_tail_corr = np.corrcoef(x[upper_tail_mask], y[upper_tail_mask])[0, 1]
                else:
                    upper_tail_corr = np.nan

                # Lower tail: correlation in bottom 10%
                q10 = np.percentile(x, 10)
                lower_tail_mask = x <= q10
                if lower_tail_mask.sum() > 10:
                    lower_tail_corr = np.corrcoef(x[lower_tail_mask], y[lower_tail_mask])[0, 1]
                else:
                    lower_tail_corr = np.nan

                # Determine dependency type
                if abs(spearman_rho - pearson_r) < 0.1:
                    dep_type = "Linear"
                elif abs(spearman_rho) > abs(pearson_r) + 0.1:
                    dep_type = "Monotonic (Non-linear)"
                else:
                    dep_type = "Non-monotonic"

                if abs(spearman_rho) > 0.5 or abs(kendall_tau) > 0.3:
                    logger.info(f"    {var1} ↔ {var2}:")
                    logger.info(f"        Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4f})")
                    logger.info(f"        Kendall τ: {kendall_tau:.4f} (p={kendall_p:.4f})")
                    logger.info(f"        Pearson r: {pearson_r:.4f}")
                    logger.info(f"        Dependency type: {dep_type}")
                    if not np.isnan(upper_tail_corr):
                        logger.info(f"        Upper tail dependency: {upper_tail_corr:.4f}")
                    if not np.isnan(lower_tail_corr):
                        logger.info(f"        Lower tail dependency: {lower_tail_corr:.4f}")

                copula_results.append(
                    {
                        "Var1": var1,
                        "Var2": var2,
                        "Pearson_r": pearson_r,
                        "Spearman_ρ": spearman_rho,
                        "Spearman_p": spearman_p,
                        "Kendall_τ": kendall_tau,
                        "Kendall_p": kendall_p,
                        "Upper_Tail_Dep": upper_tail_corr,
                        "Lower_Tail_Dep": lower_tail_corr,
                        "Dependency_Type": dep_type,
                        "Strength": "Strong"
                        if abs(spearman_rho) > 0.5
                        else "Moderate"
                        if abs(spearman_rho) > 0.3
                        else "Weak",
                    }
                )

        copula_df = pd.DataFrame(copula_results).sort_values("Spearman_ρ", key=abs, ascending=False)

        logger.info("\n    COPULA DEPENDENCY RESULTS - TOP 15 PAIRS:")
        display_copula = copula_df.head(15).copy()
        display_copula = display_copula.round(4)
        print(tabulate(display_copula, headers="keys", tablefmt="grid", showindex=False))

        # Summary by dependency type
        logger.info("\n    DEPENDENCY TYPE SUMMARY:")
        dep_type_counts = copula_df["Dependency_Type"].value_counts()
        for dep_type, count in dep_type_counts.items():
            logger.info(f"        {dep_type}: {count} pairs ({count / len(copula_df) * 100:.1f}%)")

        # Summary by strength
        logger.info("\n    DEPENDENCY STRENGTH SUMMARY:")
        strength_counts = copula_df["Strength"].value_counts()
        for strength, count in strength_counts.items():
            logger.info(f"        {strength}: {count} pairs ({count / len(copula_df) * 100:.1f}%)")

    logger.info("\n✓ Advanced dependency analysis complete!")
    logger.info("=" * 100)

    return entropy_df, mi_matrix, copula_df
