#!/usr/bin/env python3
"""
Train bet builder outcome prediction model.

This script trains a model to predict whether bet builders will hit based on
builder characteristics and features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, setup_logging, get_logger, model_config
from src.models.bet_builder_outcome import (
    default_feature_columns,
    train_builder_hit_model,
    evaluate_model,
    analyze_feature_impact,
)


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train bet builder outcome model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "bet_builders.parquet",
        help="Input bet builders file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Output directory for trained model",
    )

    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=None,
        help="Feature columns to use (default: use config defaults)",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="builder_hit",
        help="Label column name",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=model_config.random_state,
        help="Random state for reproducibility",
    )

    parser.add_argument(
        "--analyze-features",
        action="store_true",
        default=True,
        help="Perform feature importance analysis",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="builder_hit_model",
        help="Model file name (without extension)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=getattr(__import__("logging"), log_level), log_file=args.log_file)

    logger.info("=" * 80)
    logger.info("BET BUILDER MODEL TRAINING")
    logger.info("=" * 80)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load builders
        logger.info(f"\nLoading builders from {args.input_file}...")
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")

        builders = pd.read_parquet(args.input_file)
        logger.info(f"Loaded {len(builders):,} builders")

        # Determine features
        feature_cols = args.features if args.features else default_feature_columns()
        logger.info(f"\nUsing {len(feature_cols)} features:")
        for feat in feature_cols:
            logger.info(f"  - {feat}")

        # Check label distribution
        if args.label in builders.columns:
            hit_rate = builders[args.label].mean()
            logger.info(f"\nLabel distribution: {hit_rate*100:.2f}% positive class")
        else:
            raise ValueError(f"Label column '{args.label}' not found")

        # Train model
        logger.info("\nTraining model...")
        model, metrics = train_builder_hit_model(
            builders,
            feature_cols=feature_cols,
            label_col=args.label,
            random_state=args.random_state,
            verbose=True,
        )

        # Save model
        model_path = args.output_dir / f"{args.model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"\n Saved model to {model_path}")

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        # Remove non-serializable columns
        if "feature_importance" in metrics_df.columns:
            feature_importance = metrics_df["feature_importance"].iloc[0]
            metrics_df = metrics_df.drop(columns=["feature_importance"])

            # Save feature importance separately
            feat_imp_path = args.output_dir / f"{args.model_name}_feature_importance.parquet"
            feature_importance.to_parquet(feat_imp_path, index=False)
            logger.info(f" Saved feature importance to {feat_imp_path}")

        metrics_path = args.output_dir / f"{args.model_name}_metrics.parquet"
        metrics_df.to_parquet(metrics_path, index=False)
        logger.info(f" Saved metrics to {metrics_path}")

        # Feature analysis
        if args.analyze_features:
            logger.info("\nAnalyzing feature impact...")
            feature_analysis = analyze_feature_impact(
                model,
                builders,
                feature_cols=feature_cols,
                n_top=10,
            )

            analysis_path = args.output_dir / f"{args.model_name}_feature_analysis.parquet"
            feature_analysis.to_parquet(analysis_path, index=False)
            logger.info(f" Saved feature analysis to {analysis_path}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training samples: {metrics['n_obs']:,}")
        logger.info(f"Features: {metrics['n_features']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Brier Score: {metrics['brier']:.4f}")
        logger.info(f"Log Loss: {metrics['log_loss']:.4f}")

        logger.info("\n Model training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
