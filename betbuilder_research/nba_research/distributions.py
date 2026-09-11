"""Preserved research calculations: distributions."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.config import get_logger
from .plotting import dataset_title

logger = get_logger(__name__)


def test_distributions(
    df: pd.DataFrame, output_dir: Path | None = None, plot: bool = True
) -> pd.DataFrame:
    """
    Test each numeric variable against multiple distributions to find best fit.
    Uses Kolmogorov-Smirnov, Anderson-Darling, and Chi-Square tests.

    Returns DataFrame with distribution test results for each variable.
    """
    from scipy import stats
    from scipy.stats import kstest, anderson, jarque_bera

    logger.info("\n" + "=" * 100)
    logger.info("DISTRIBUTION TESTING - STATISTICAL GOODNESS-OF-FIT ANALYSIS")
    logger.info("=" * 100)

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Distributions to test
    distributions_to_test = {
        "Normal": stats.norm,
        "Lognormal": stats.lognorm,
        "Exponential": stats.expon,
        "Gamma": stats.gamma,
        "Beta": stats.beta,
        "Weibull": stats.weibull_min,
        "Poisson": stats.poisson,
        "Negative Binomial": stats.nbinom,
    }

    logger.info(
        f"\n>>> Testing {len(numeric_cols)} variables against {len(distributions_to_test)} distributions..."
    )
    logger.info("    Using: Kolmogorov-Smirnov, Anderson-Darling, Shapiro-Wilk, Jarque-Bera tests")

    results = []

    if output_dir is None:
        output_dir = Path("output/figures")
    dist_plot_dir = output_dir / "distribution_fits"
    if plot:
        dist_plot_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_cols:
        data = df[col].dropna()

        if len(data) < 3:
            continue

        logger.info(f"\n>>> Testing variable: {col}")
        logger.info(f"    Sample size: n={len(data):,}")
        logger.info(f"    Range: [{data.min():.3f}, {data.max():.3f}]")
        logger.info(f"    Mean: {data.mean():.3f}, Std: {data.std():.3f}")
        logger.info(f"    Skewness: {stats.skew(data):.3f}, Kurtosis: {stats.kurtosis(data):.3f}")

        best_dist = None
        best_ks_stat = np.inf
        best_ks_pval = 0

        dist_results = {}

        for dist_name, dist_func in distributions_to_test.items():
            try:
                # Skip distributions not suitable for the data
                if dist_name == "Lognormal" and (data <= 0).any():
                    continue
                if dist_name == "Beta" and ((data < 0).any() or (data > 1).any()):
                    continue
                if dist_name in ["Poisson", "Negative Binomial"] and not np.allclose(
                    data, data.astype(int)
                ):
                    continue

                # Fit distribution
                if dist_name == "Normal":
                    params = dist_func.fit(data)
                    fitted = dist_func(*params)
                elif dist_name == "Exponential":
                    params = dist_func.fit(data, floc=0)
                    fitted = dist_func(*params)
                elif dist_name == "Poisson":
                    params = (data.mean(),)
                    fitted = dist_func(*params)
                else:
                    params = dist_func.fit(data)
                    fitted = dist_func(*params)

                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = kstest(data, lambda x: fitted.cdf(x))

                # Anderson-Darling test (only for normal, exponential, logistic)
                ad_stat = None
                ad_critical = None
                if dist_name in ["Normal", "Exponential"]:
                    ad_result = anderson(data, dist=dist_name.lower())
                    ad_stat = ad_result.statistic
                    ad_critical = ad_result.critical_values[2]  # 5% significance level

                dist_results[dist_name] = {
                    "ks_stat": ks_stat,
                    "ks_pval": ks_pval,
                    "ad_stat": ad_stat,
                    "ad_critical": ad_critical,
                    "params": params,
                }

                # Track best fit (lowest KS statistic, highest p-value)
                if ks_pval > best_ks_pval:
                    best_ks_pval = ks_pval
                    best_ks_stat = ks_stat
                    best_dist = dist_name

            except Exception as e:
                logger.warning(f"        {col}/{dist_name}: candidate unavailable ({str(e)[:100]})")
                continue

        # Normality tests
        shapiro_stat, shapiro_pval = (
            stats.shapiro(data[:5000]) if len(data) > 5000 else stats.shapiro(data)
        )
        jb_stat, jb_pval = jarque_bera(data)

        # Determine best distribution
        if best_dist:
            logger.info(f"    BEST FIT: {best_dist}")
            logger.info(f"        KS statistic: {best_ks_stat:.4f}")
            logger.info(f"        KS p-value: {best_ks_pval:.4f}")
            if dist_results[best_dist]["ad_stat"]:
                logger.info(f"        AD statistic: {dist_results[best_dist]['ad_stat']:.4f}")

        logger.info("    NORMALITY TESTS:")
        logger.info(f"        Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_pval:.4f}")
        logger.info(f"        Jarque-Bera: JB={jb_stat:.4f}, p={jb_pval:.4f}")

        # Interpretation
        is_normal = (shapiro_pval > 0.05) and (jb_pval > 0.05)
        interpretation = "NORMAL" if is_normal else "NON-NORMAL"
        logger.info(f"    INTERPRETATION: {interpretation} (α=0.05)")

        results.append(
            {
                "Variable": col,
                "N": len(data),
                "Mean": data.mean(),
                "Std": data.std(),
                "Skewness": stats.skew(data),
                "Kurtosis": stats.kurtosis(data),
                "Best_Distribution": best_dist if best_dist else "Unknown",
                "KS_Statistic": best_ks_stat if best_ks_stat != np.inf else np.nan,
                "KS_P_Value": best_ks_pval,
                "Shapiro_W": shapiro_stat,
                "Shapiro_P": shapiro_pval,
                "JB_Statistic": jb_stat,
                "JB_P_Value": jb_pval,
                "Is_Normal": "Yes" if is_normal else "No",
            }
        )

        # Plot all fitted distributions for this variable
        if plot and len(dist_results) > 0:
            try:
                fig, ax = plt.subplots(figsize=(12, 7))

                # Histogram of data
                ax.hist(
                    data,
                    bins=50,
                    density=True,
                    alpha=0.35,
                    color="gray",
                    edgecolor="white",
                    label="Data",
                )

                x_min, x_max = float(np.min(data)), float(np.max(data))
                x_grid = np.linspace(x_min, x_max, 400)

                # Plot in a fixed order for readability
                for dist_name in distributions_to_test.keys():
                    if dist_name not in dist_results:
                        continue
                    params = dist_results[dist_name]["params"]
                    dist_func = distributions_to_test[dist_name]

                    if dist_name in ["Poisson", "Negative Binomial"]:
                        x_int = np.unique(
                            np.linspace(
                                int(np.floor(x_min)),
                                int(np.ceil(x_max)),
                                min(2000, int(np.ceil(x_max) - np.floor(x_min)) + 1),
                            ).astype(int)
                        )
                        if dist_name == "Poisson":
                            y = dist_func.pmf(x_int, *params)
                        else:
                            y = dist_func.pmf(x_int, *params)
                        ax.plot(x_int, y, linewidth=2, label=dist_name)
                    else:
                        y = dist_func.pdf(x_grid, *params)
                        ax.plot(x_grid, y, linewidth=2, label=dist_name)

                ax.set_title(dataset_title(f"Distribution Fits - {col}"), fontweight="bold")
                ax.set_xlabel(col)
                ax.set_ylabel("Density")
                ax.legend(fontsize=8, ncol=2)
                ax.grid(True, alpha=0.2)

                fig.tight_layout()
                fig_path = dist_plot_dir / f"distribution_fits_{col}.png"
                plt.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Plot failed for {col}: {str(e)[:80]}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    logger.info("\n" + "=" * 100)
    logger.info("DISTRIBUTION TESTING RESULTS - SUMMARY TABLE")
    logger.info("=" * 100)

    # Display full results
    display_df = results_df.copy()
    display_df = display_df.round(4)
    print("\n" + tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))

    # Summary statistics
    logger.info("\n>>> DISTRIBUTION SUMMARY:")
    dist_counts = results_df["Best_Distribution"].value_counts()
    for dist, count in dist_counts.items():
        logger.info(f"    {dist}: {count} variables ({count / len(results_df) * 100:.1f}%)")

    normal_count = (results_df["Is_Normal"] == "Yes").sum()
    logger.info(
        f"\n    Variables passing normality tests: {normal_count}/{len(results_df)} ({normal_count / len(results_df) * 100:.1f}%)"
    )

    logger.info("\n✓ Distribution testing complete!")

    return results_df
