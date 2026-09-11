"""
Data Diagnostics Module.

Automated data health checks performed at every pipeline stage.
Detects issues like zero-variance, outliers, duplicates, and missingness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..config import get_logger
from .table_formatter import print_warning_block, print_table


logger = get_logger(__name__)


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    check_name: str
    passed: bool
    severity: str  # "info", "warning", "critical"
    message: str
    details: Optional[Dict[str, Any]] = None
    sample_df: Optional[pd.DataFrame] = None


@dataclass
class HealthReport:
    """Complete health report for a dataset."""
    dataset_name: str
    row_count: int
    col_count: int
    memory_mb: float
    results: List[DiagnosticResult] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity in ["warning", "critical"] for r in self.results)

    @property
    def has_critical(self) -> bool:
        return any(r.severity == "critical" for r in self.results)

    @property
    def warnings(self) -> List[DiagnosticResult]:
        return [r for r in self.results if r.severity == "warning"]

    @property
    def critical(self) -> List[DiagnosticResult]:
        return [r for r in self.results if r.severity == "critical"]


class DataDiagnostics:
    """
    Automated data health diagnostics engine.

    Performs comprehensive checks on DataFrames and reports issues.
    """

    def __init__(
        self,
        outlier_threshold: float = 5.0,  # IQR multiplier
        missing_warning_pct: float = 5.0,
        missing_critical_pct: float = 20.0,
        duplicate_warning_pct: float = 1.0,
    ):
        self.outlier_threshold = outlier_threshold
        self.missing_warning_pct = missing_warning_pct
        self.missing_critical_pct = missing_critical_pct
        self.duplicate_warning_pct = duplicate_warning_pct
        self._reports: List[HealthReport] = []

    def run_all_checks(
        self,
        df: pd.DataFrame,
        name: str,
        key_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        print_warnings: bool = True,
    ) -> HealthReport:
        """
        Run all diagnostic checks on a DataFrame.

        Args:
            df: DataFrame to check
            name: Name of the dataset
            key_columns: Columns to check for duplicates
            numeric_columns: Columns to check for outliers
            print_warnings: Whether to print warnings to console

        Returns:
            HealthReport with all diagnostic results
        """
        report = HealthReport(
            dataset_name=name,
            row_count=len(df),
            col_count=len(df.columns),
            memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
        )

        # Run checks
        report.results.append(self.check_zero_variance(df))
        report.results.append(self.check_missing_values(df))

        if key_columns:
            report.results.append(self.check_duplicates(df, key_columns))

        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    result = self.check_outliers(df, col)
                    if not result.passed:
                        report.results.append(result)

        report.results.append(self.check_dtype_consistency(df))
        report.results.append(self.check_constant_columns(df))

        # Print warnings if requested
        if print_warnings and report.has_warnings:
            warnings = [r.message for r in report.results if r.severity in ["warning", "critical"]]
            sample_dfs = [r.sample_df for r in report.results if r.sample_df is not None]
            sample = sample_dfs[0] if sample_dfs else None

            print_warning_block(
                warnings,
                title=f"DATA HEALTH WARNINGS: {name}",
                sample_df=sample
            )

        self._reports.append(report)
        return report

    def check_zero_variance(self, df: pd.DataFrame) -> DiagnosticResult:
        """Check for zero-variance columns."""
        zero_var = []

        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                zero_var.append(col)

        if zero_var:
            return DiagnosticResult(
                check_name="zero_variance",
                passed=False,
                severity="warning",
                message=f"Zero-variance columns detected: {', '.join(zero_var[:5])}{'...' if len(zero_var) > 5 else ''}",
                details={"columns": zero_var},
            )

        return DiagnosticResult(
            check_name="zero_variance",
            passed=True,
            severity="info",
            message="No zero-variance columns detected",
        )

    def check_missing_values(self, df: pd.DataFrame) -> DiagnosticResult:
        """Check for missing values."""
        missing_pct = df.isna().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > self.missing_warning_pct]

        if len(high_missing) == 0:
            return DiagnosticResult(
                check_name="missing_values",
                passed=True,
                severity="info",
                message="No significant missing values",
            )

        critical = high_missing[high_missing > self.missing_critical_pct]
        severity = "critical" if len(critical) > 0 else "warning"

        # Create sample of missing
        top_missing = high_missing.nlargest(5)
        details_str = ", ".join([f"{col}: {pct:.1f}%" for col, pct in top_missing.items()])

        return DiagnosticResult(
            check_name="missing_values",
            passed=False,
            severity=severity,
            message=f"High missingness detected: {details_str}",
            details={"missing_pct": high_missing.to_dict()},
        )

    def check_duplicates(
        self,
        df: pd.DataFrame,
        key_columns: List[str]
    ) -> DiagnosticResult:
        """Check for duplicate keys."""
        valid_keys = [k for k in key_columns if k in df.columns]
        if not valid_keys:
            return DiagnosticResult(
                check_name="duplicates",
                passed=True,
                severity="info",
                message="No key columns specified for duplicate check",
            )

        duplicates = df[df.duplicated(subset=valid_keys, keep=False)]
        dup_pct = len(duplicates) / len(df) * 100 if len(df) > 0 else 0

        if dup_pct < self.duplicate_warning_pct:
            return DiagnosticResult(
                check_name="duplicates",
                passed=True,
                severity="info",
                message=f"Duplicate rate acceptable: {dup_pct:.2f}%",
            )

        return DiagnosticResult(
            check_name="duplicates",
            passed=False,
            severity="warning",
            message=f"Duplicate keys detected: {dup_pct:.1f}% of rows on columns {valid_keys}",
            details={"duplicate_count": len(duplicates), "duplicate_pct": dup_pct},
            sample_df=duplicates.head(10),
        )

    def check_outliers(
        self,
        df: pd.DataFrame,
        column: str
    ) -> DiagnosticResult:
        """Check for extreme outliers using IQR method."""
        if column not in df.columns:
            return DiagnosticResult(
                check_name=f"outliers_{column}",
                passed=True,
                severity="info",
                message=f"Column {column} not found",
            )

        data = df[column].dropna()
        if len(data) == 0:
            return DiagnosticResult(
                check_name=f"outliers_{column}",
                passed=True,
                severity="info",
                message=f"Column {column} is empty",
            )

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_pct = len(outliers) / len(df) * 100 if len(df) > 0 else 0

        if outlier_pct < 0.5:  # Less than 0.5% outliers is acceptable
            return DiagnosticResult(
                check_name=f"outliers_{column}",
                passed=True,
                severity="info",
                message=f"Outliers in {column}: {outlier_pct:.2f}% (acceptable)",
            )

        return DiagnosticResult(
            check_name=f"outliers_{column}",
            passed=False,
            severity="warning",
            message=f"Extreme outliers in {column}: {outlier_pct:.1f}% outside [{lower_bound:.1f}, {upper_bound:.1f}]",
            details={
                "outlier_count": len(outliers),
                "outlier_pct": outlier_pct,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "min": data.min(),
                "max": data.max(),
            },
            sample_df=outliers.head(10) if len(outliers) > 0 else None,
        )

    def check_dtype_consistency(self, df: pd.DataFrame) -> DiagnosticResult:
        """Check for mixed dtypes in object columns."""
        mixed_dtype_cols = []

        for col in df.columns:
            if df[col].dtype == object:
                # Check if mixed types
                types = df[col].dropna().apply(type).unique()
                if len(types) > 1:
                    mixed_dtype_cols.append(col)

        if not mixed_dtype_cols:
            return DiagnosticResult(
                check_name="dtype_consistency",
                passed=True,
                severity="info",
                message="All columns have consistent dtypes",
            )

        return DiagnosticResult(
            check_name="dtype_consistency",
            passed=False,
            severity="warning",
            message=f"Mixed dtypes in columns: {', '.join(mixed_dtype_cols[:5])}",
            details={"columns": mixed_dtype_cols},
        )

    def check_constant_columns(self, df: pd.DataFrame) -> DiagnosticResult:
        """Check for constant (single-value) columns."""
        constant_cols = []

        for col in df.columns:
            if df[col].nunique(dropna=True) == 1:
                constant_cols.append(col)

        if not constant_cols:
            return DiagnosticResult(
                check_name="constant_columns",
                passed=True,
                severity="info",
                message="No constant columns detected",
            )

        return DiagnosticResult(
            check_name="constant_columns",
            passed=False,
            severity="warning",
            message=f"Constant columns (may be stuck flags): {', '.join(constant_cols[:5])}",
            details={"columns": constant_cols},
        )

    def get_all_reports(self) -> List[HealthReport]:
        """Get all health reports generated during this session."""
        return self._reports

    def clear_reports(self) -> None:
        """Clear report history."""
        self._reports = []


# Convenience functions

_default_diagnostics = DataDiagnostics()


def run_data_health_checks(
    df: pd.DataFrame,
    name: str,
    key_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    print_warnings: bool = True,
) -> HealthReport:
    """Run all data health checks on a DataFrame."""
    return _default_diagnostics.run_all_checks(
        df, name, key_columns, numeric_columns, print_warnings
    )


def check_zero_variance(df: pd.DataFrame) -> List[str]:
    """Return list of zero-variance column names."""
    result = _default_diagnostics.check_zero_variance(df)
    return result.details.get("columns", []) if result.details else []


def check_duplicates(df: pd.DataFrame, key_columns: List[str]) -> float:
    """Return duplicate percentage for given key columns."""
    result = _default_diagnostics.check_duplicates(df, key_columns)
    return result.details.get("duplicate_pct", 0.0) if result.details else 0.0


def check_outliers(df: pd.DataFrame, column: str) -> Tuple[int, float, float]:
    """Return (outlier_count, lower_bound, upper_bound) for a column."""
    result = _default_diagnostics.check_outliers(df, column)
    if result.details:
        return (
            result.details.get("outlier_count", 0),
            result.details.get("lower_bound", 0),
            result.details.get("upper_bound", 0),
        )
    return (0, 0, 0)
