"""Model for predicting bet builder outcomes."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import get_logger, model_config


logger = get_logger(__name__)


def default_feature_columns() -> List[str]:
    """Get default feature columns for builder outcome model."""
    return model_config.feature_cols


def train_builder_hit_model(
    builders: pd.DataFrame,
    feature_cols: List[str] | None = None,
    label_col: str = "builder_hit",
    random_state: int | None = None,
    verbose: bool = True,
) -> Tuple[Pipeline, dict]:
    """
    Train a logistic regression model to predict builder success.

    Args:
        builders: DataFrame with builder-level features
        feature_cols: List of feature column names
        label_col: Name of target column
        random_state: Random seed
        verbose: Whether to log detailed metrics

    Returns:
        Tuple of (fitted pipeline, metrics dict)
    """
    feature_cols = feature_cols or default_feature_columns()
    random_state = random_state or model_config.random_state

    logger.info(f"Training builder hit model with {len(feature_cols)} features")

    # Validate features exist
    missing_features = set(feature_cols) - set(builders.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    X = builders[feature_cols].to_numpy()
    y = builders[label_col].to_numpy()

    logger.info(f"Training on {len(X):,} observations")
    logger.info(f"Positive class rate: {y.mean()*100:.2f}%")

    # Build pipeline
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced"  # Handle class imbalance
            )),
        ]
    )

    # Fit model
    logger.info("Fitting model...")
    pipeline.fit(X, y)

    # Get feature importance (coefficients)
    coefs = pipeline.named_steps["clf"].coef_[0]
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    if verbose:
        logger.info("Top 5 most important features:")
        for _, row in feature_importance.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

    # In-sample diagnostics
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "brier": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-7, 1 - 1e-7))),
        "roc_auc": float(roc_auc_score(y, probs)),
        "n_obs": int(len(y)),
        "n_features": len(feature_cols),
        "positive_rate": float(y.mean()),
    }

    if verbose:
        logger.info("In-sample performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y, preds)
        logger.info(f"  Confusion Matrix:\n{cm}")

    # Store feature importance
    metrics["feature_importance"] = feature_importance

    return pipeline, metrics


def predict_builder_outcomes(
    model: Pipeline,
    builders: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Predict outcomes for bet builders.

    Args:
        model: Trained model pipeline
        builders: DataFrame with builder features
        feature_cols: List of feature column names

    Returns:
        DataFrame with predictions added
    """
    feature_cols = feature_cols or default_feature_columns()

    logger.info(f"Generating predictions for {len(builders):,} builders")

    X = builders[feature_cols].to_numpy()

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = builders.copy()
    result["pred_prob"] = probs
    result["pred_hit"] = preds

    logger.info(f"Predicted hit rate: {preds.mean()*100:.2f}%")

    return result


def evaluate_model(
    model: Pipeline,
    builders: pd.DataFrame,
    feature_cols: List[str] | None = None,
    label_col: str = "builder_hit",
    verbose: bool = True,
) -> dict:
    """
    Evaluate model performance on a dataset.

    Args:
        model: Trained model pipeline
        builders: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Name of target column
        verbose: Whether to log detailed metrics

    Returns:
        Dictionary with evaluation metrics
    """
    feature_cols = feature_cols or default_feature_columns()

    logger.info(f"Evaluating model on {len(builders):,} observations")

    X = builders[feature_cols].to_numpy()
    y = builders[label_col].to_numpy()

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "brier": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-7, 1 - 1e-7))),
        "roc_auc": float(roc_auc_score(y, probs)),
        "n_obs": int(len(y)),
        "positive_rate": float(y.mean()),
        "pred_positive_rate": float(preds.mean()),
    }

    if verbose:
        logger.info("Model performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier Score: {metrics['brier']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"  True positive rate: {metrics['positive_rate']*100:.2f}%")
        logger.info(f"  Pred positive rate: {metrics['pred_positive_rate']*100:.2f}%")

        # Classification report
        logger.info("Classification report:")
        logger.info(f"\n{classification_report(y, preds, zero_division=0)}")

    return metrics


def analyze_feature_impact(
    model: Pipeline,
    builders: pd.DataFrame,
    feature_cols: List[str] | None = None,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Analyze the impact of each feature on predictions.

    Args:
        model: Trained model pipeline
        builders: DataFrame with features
        feature_cols: List of feature column names
        n_top: Number of top features to return

    Returns:
        DataFrame with feature analysis
    """
    feature_cols = feature_cols or default_feature_columns()

    logger.info("Analyzing feature impact...")

    # Get coefficients
    coefs = model.named_steps["clf"].coef_[0]

    # Get feature statistics from data
    stats = builders[feature_cols].describe().T

    # Combine
    feature_analysis = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
    })

    # Sort by absolute coefficient
    feature_analysis = feature_analysis.sort_values(
        "abs_coefficient",
        ascending=False
    ).reset_index(drop=True)

    logger.info(f"Top {n_top} features by importance:")
    for _, row in feature_analysis.head(n_top).iterrows():
        direction = "increases" if row["coefficient"] > 0 else "decreases"
        logger.info(
            f"  {row['feature']}: coef={row['coefficient']:.4f} "
            f"({direction} prob of hit)"
        )

    return feature_analysis


def calibration_analysis(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Perform calibration analysis on predictions.

    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        DataFrame with calibration statistics
    """
    from sklearn.calibration import calibration_curve

    logger.info("Performing calibration analysis...")

    # Compute calibration curve
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_pred_probs,
        n_bins=n_bins,
        strategy="quantile"
    )

    # Create DataFrame
    calibration_df = pd.DataFrame({
        "predicted_prob": mean_pred,
        "actual_rate": frac_pos,
        "calibration_error": frac_pos - mean_pred,
    })

    # Log results
    logger.info("Calibration results:")
    for _, row in calibration_df.iterrows():
        logger.info(
            f"  Predicted: {row['predicted_prob']:.3f}, "
            f"Actual: {row['actual_rate']:.3f}, "
            f"Error: {row['calibration_error']:.3f}"
        )

    # Overall calibration error
    mean_abs_error = calibration_df["calibration_error"].abs().mean()
    logger.info(f"Mean absolute calibration error: {mean_abs_error:.4f}")

    return calibration_df
