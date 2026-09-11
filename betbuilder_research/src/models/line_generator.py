"""
Learned Line Generator Module.

Learns how bookmakers set prop lines relative to player statistics,
then generates realistic synthetic lines for simulation.

This replaces the simple rolling average + noise approach with a
machine learning model that captures real bookmaker patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

from ..config import get_logger, line_generator_config, LineGeneratorConfig


logger = get_logger(__name__)


@dataclass
class LineGeneratorMetrics:
    """Metrics from line generator model training."""

    n_samples: int
    n_features: int
    mae: float                          # Mean Absolute Error
    rmse: float                         # Root Mean Squared Error
    r2: float                           # R-squared
    mae_by_stat: Dict[str, float]       # Per-stat MAE
    samples_by_stat: Dict[str, int]     # Samples per stat type
    model_type: str
    confidence_level: str               # "insufficient", "low", "medium", "high"
    is_reliable: bool                   # Whether sample size threshold met

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mae_by_stat": self.mae_by_stat,
            "samples_by_stat": self.samples_by_stat,
            "model_type": self.model_type,
            "confidence_level": self.confidence_level,
            "is_reliable": self.is_reliable,
        }


def validate_real_lines_csv(path: Path) -> Tuple[bool, List[str]]:
    """
    Validate format of real betting lines CSV.

    Expected columns:
        - player_id (required): Player identifier
        - stat (required): One of pts, fg3m, ast, reb
        - line (required): Float threshold value
        - game_date (required): Date of the game
        - over_odds (optional): Decimal odds for over
        - under_odds (optional): Decimal odds for under
        - book (optional): Bookmaker name

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        df = pd.read_csv(path, nrows=100)  # Sample for validation
    except Exception as e:
        return False, [f"Error reading CSV: {e}"]

    # Check required columns
    required = ["player_id", "stat", "line", "game_date"]
    missing = set(required) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Validate stat values
    if "stat" in df.columns:
        valid_stats = {"pts", "fg3m", "ast", "reb"}
        invalid_stats = set(df["stat"].unique()) - valid_stats
        if invalid_stats:
            errors.append(
                f"Invalid stat values: {invalid_stats}. "
                f"Expected: {valid_stats}"
            )

    # Validate line is numeric
    if "line" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["line"]):
            errors.append("Column 'line' must be numeric")
        elif (df["line"] < 0).any():
            errors.append("Column 'line' contains negative values")

    # Validate date format
    if "game_date" in df.columns:
        try:
            pd.to_datetime(df["game_date"])
        except Exception:
            errors.append("Column 'game_date' has invalid date format")

    is_valid = len(errors) == 0

    if is_valid:
        logger.info(f"CSV validation passed: {path}")
        logger.info(f"  Rows (sampled): {len(df)}")
        logger.info(f"  Stats found: {df['stat'].unique().tolist()}")
    else:
        logger.error(f"CSV validation failed: {path}")
        for err in errors:
            logger.error(f"  - {err}")

    return is_valid, errors


def compute_line_features(
    player_games: pd.DataFrame,
    stat_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Compute features for line prediction.

    All features use shift(1) to prevent lookahead bias.

    Features computed:
        - rolling_avg_{window}: Rolling mean for each window
        - rolling_std_{window}: Rolling std for each window
        - rolling_cv_{window}: Coefficient of variation
        - career_avg: Expanding mean
        - career_var: Expanding variance
        - games_played: Cumulative game count

    Args:
        player_games: DataFrame with player game logs
        stat_cols: Stats to compute features for
        windows: Rolling window sizes

    Returns:
        DataFrame with computed features
    """
    config = line_generator_config
    stat_cols = stat_cols or config.stat_cols
    windows = windows or config.rolling_windows

    logger.info(f"Computing line features for {len(stat_cols)} stats, windows={windows}")

    df = player_games.sort_values(["player_id", "game_date"]).copy()

    for stat in stat_cols:
        if stat not in df.columns:
            logger.warning(f"Stat column '{stat}' not found, skipping")
            continue

        for window in windows:
            # Rolling mean (shifted to exclude current game)
            df[f"{stat}_rolling_avg_{window}"] = (
                df.groupby("player_id")[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
            )

            # Rolling std
            df[f"{stat}_rolling_std_{window}"] = (
                df.groupby("player_id")[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
            )

        # Coefficient of variation (using 10-game window)
        primary_window = 10 if 10 in windows else windows[0]
        avg_col = f"{stat}_rolling_avg_{primary_window}"
        std_col = f"{stat}_rolling_std_{primary_window}"

        if avg_col in df.columns and std_col in df.columns:
            df[f"{stat}_cv"] = (
                df[std_col] / df[avg_col].replace(0, np.nan)
            )

        # Career stats (expanding window, shifted)
        df[f"{stat}_career_avg"] = (
            df.groupby("player_id")[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
        )
        df[f"{stat}_career_var"] = (
            df.groupby("player_id")[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=5).var())
        )

    # Games played count per player
    df["games_played"] = df.groupby("player_id").cumcount()

    logger.info(f"Computed features for {len(df)} rows")

    return df


def _get_confidence_level(n_samples: int, config: LineGeneratorConfig) -> str:
    """
    Determine confidence level based on sample size.

    Returns:
        "insufficient": Cannot train reliably
        "low": Can train but high variance expected
        "medium": Reasonable reliability
        "high": Production ready
    """
    if n_samples < config.min_samples_train:
        return "insufficient"
    elif n_samples < config.min_samples_reliable:
        return "low"
    elif n_samples < 5000:
        return "medium"
    else:
        return "high"


class LearnedLineGenerator:
    """
    Machine learning model to predict betting lines from player statistics.

    Learns the relationship:
        line = f(rolling_avg, variance, stat_type, player_factors)

    Captures:
        - How lines relate to rolling averages
        - Stat-specific adjustments (pts lines set differently than fg3m)
        - Rounding patterns to 0.5 increments
        - Variance-based adjustments

    Example usage:
        >>> generator = LearnedLineGenerator()
        >>> metrics = generator.fit(player_games, real_lines)
        >>> print(f"MAE: {metrics.mae:.2f}")
        >>> synthetic_lines = generator.generate_lines(player_games)
    """

    def __init__(self, config: Optional[LineGeneratorConfig] = None):
        """
        Initialize the line generator.

        Args:
            config: Configuration (defaults to global line_generator_config)
        """
        self.config = config or line_generator_config
        self.model: Optional[Pipeline] = None
        self.feature_cols: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.is_fitted: bool = False
        self.metrics: Optional[LineGeneratorMetrics] = None
        self._stat_encoder: Optional[Dict[str, int]] = None

    def _build_feature_matrix(
        self,
        player_games_with_features: pd.DataFrame,
        lines_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix by merging features with lines.

        Args:
            player_games_with_features: Player games with computed features
            lines_df: Real betting lines

        Returns:
            Tuple of (feature_df, target_series)
        """
        # Ensure consistent date format
        player_games_with_features = player_games_with_features.copy()
        lines_df = lines_df.copy()

        player_games_with_features["game_date"] = pd.to_datetime(
            player_games_with_features["game_date"]
        ).dt.date
        lines_df["game_date"] = pd.to_datetime(lines_df["game_date"]).dt.date

        # Melt player games to long format (one row per player-game-stat)
        feature_cols_for_merge = []
        for stat in self.config.stat_cols:
            stat_features = [
                col for col in player_games_with_features.columns
                if col.startswith(f"{stat}_")
            ]
            feature_cols_for_merge.extend(stat_features)

        # Merge lines with features
        merged = lines_df.merge(
            player_games_with_features[
                ["player_id", "game_date", "games_played"] + feature_cols_for_merge
            ],
            on=["player_id", "game_date"],
            how="inner"
        )

        if len(merged) == 0:
            logger.warning("No matches found between lines and player games")
            return pd.DataFrame(), pd.Series(dtype=float)

        logger.info(f"Merged {len(merged)} lines with features")

        # Build feature matrix for each row
        feature_records = []
        targets = []

        for _, row in merged.iterrows():
            stat = row["stat"]
            line = row["line"]

            # Get stat-specific features
            features = {
                "stat_type": stat,
                "games_played": row.get("games_played", 0),
            }

            # Add rolling features for this stat
            for window in self.config.rolling_windows:
                avg_col = f"{stat}_rolling_avg_{window}"
                std_col = f"{stat}_rolling_std_{window}"

                features[f"rolling_avg_{window}"] = row.get(avg_col, np.nan)
                features[f"rolling_std_{window}"] = row.get(std_col, np.nan)

            # Add career features
            features["career_avg"] = row.get(f"{stat}_career_avg", np.nan)
            features["career_var"] = row.get(f"{stat}_career_var", np.nan)
            features["cv"] = row.get(f"{stat}_cv", np.nan)

            feature_records.append(features)
            targets.append(line)

        feature_df = pd.DataFrame(feature_records)
        target_series = pd.Series(targets, name="line")

        return feature_df, target_series

    def _build_pipeline(self, model_type: str) -> Pipeline:
        """
        Build sklearn pipeline based on model type.

        Args:
            model_type: "gradient_boosting" or "linear"

        Returns:
            sklearn Pipeline
        """
        # Numeric features
        self.numeric_features = [
            f"rolling_avg_{w}" for w in self.config.rolling_windows
        ] + [
            f"rolling_std_{w}" for w in self.config.rolling_windows
        ] + ["career_avg", "career_var", "cv", "games_played"]

        # Categorical features
        self.categorical_features = ["stat_type"]

        # Preprocessor
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            remainder="drop"
        )

        # Select regressor
        if model_type == "gradient_boosting":
            regressor = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )
        else:  # linear
            regressor = LinearRegression()

        # Full pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        return pipeline

    def fit(
        self,
        player_games: pd.DataFrame,
        real_lines: pd.DataFrame,
        verbose: bool = True
    ) -> LineGeneratorMetrics:
        """
        Train the line generator on real betting lines.

        Args:
            player_games: Historical player game logs
            real_lines: Real betting lines with columns:
                - player_id, stat, line, game_date
            verbose: Whether to log progress

        Returns:
            Training metrics
        """
        logger.info("=" * 60)
        logger.info("Training Learned Line Generator")
        logger.info("=" * 60)

        # Compute features
        player_games_features = compute_line_features(
            player_games,
            stat_cols=self.config.stat_cols,
            windows=self.config.rolling_windows
        )

        # Build feature matrix
        X, y = self._build_feature_matrix(player_games_features, real_lines)

        if len(X) == 0:
            raise ValueError("No valid training data after merging lines with features")

        n_samples = len(X)
        confidence = _get_confidence_level(n_samples, self.config)

        logger.info(f"Training samples: {n_samples}")
        logger.info(f"Confidence level: {confidence}")

        # Check sample threshold
        if confidence == "insufficient":
            logger.warning(
                f"Insufficient samples ({n_samples} < {self.config.min_samples_train}). "
                f"Consider collecting more data."
            )

        # Determine model type based on samples
        if n_samples < self.config.min_samples_train:
            model_type = "linear"
            logger.info("Using linear regression due to small sample size")
        else:
            model_type = self.config.model_type
            logger.info(f"Using {model_type} model")

        # Build and fit pipeline
        self.model = self._build_pipeline(model_type)

        logger.info("Fitting model...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Compute metrics
        y_pred = self.model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Per-stat metrics
        mae_by_stat = {}
        samples_by_stat = {}

        for stat in self.config.stat_cols:
            stat_mask = X["stat_type"] == stat
            if stat_mask.sum() > 0:
                stat_mae = mean_absolute_error(y[stat_mask], y_pred[stat_mask])
                mae_by_stat[stat] = float(stat_mae)
                samples_by_stat[stat] = int(stat_mask.sum())

        self.metrics = LineGeneratorMetrics(
            n_samples=n_samples,
            n_features=X.shape[1],
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            mae_by_stat=mae_by_stat,
            samples_by_stat=samples_by_stat,
            model_type=model_type,
            confidence_level=confidence,
            is_reliable=n_samples >= self.config.min_samples_reliable,
        )

        if verbose:
            logger.info("-" * 40)
            logger.info("Training Results:")
            logger.info(f"  MAE: {mae:.3f} (average line error)")
            logger.info(f"  RMSE: {rmse:.3f}")
            logger.info(f"  R2: {r2:.3f}")
            logger.info("  Per-stat MAE:")
            for stat, stat_mae in mae_by_stat.items():
                logger.info(f"    {stat}: {stat_mae:.3f} ({samples_by_stat[stat]} samples)")
            logger.info(f"  Reliable: {self.metrics.is_reliable}")
            logger.info("-" * 40)

        return self.metrics

    def predict_line(
        self,
        stat: str,
        rolling_avgs: Dict[int, float],
        rolling_stds: Dict[int, float],
        career_avg: float,
        career_var: float,
        cv: float,
        games_played: int,
    ) -> float:
        """
        Predict a single line value.

        Args:
            stat: Stat type (pts, ast, reb, fg3m)
            rolling_avgs: Dict mapping window -> rolling average
            rolling_stds: Dict mapping window -> rolling std
            career_avg: Career average
            career_var: Career variance
            cv: Coefficient of variation
            games_played: Number of games played

        Returns:
            Predicted line value (rounded to 0.5)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Build feature dict
        features = {
            "stat_type": stat,
            "games_played": games_played,
            "career_avg": career_avg,
            "career_var": career_var,
            "cv": cv,
        }

        for window in self.config.rolling_windows:
            features[f"rolling_avg_{window}"] = rolling_avgs.get(window, np.nan)
            features[f"rolling_std_{window}"] = rolling_stds.get(window, np.nan)

        X = pd.DataFrame([features])
        raw_pred = self.model.predict(X)[0]

        # Round to nearest increment
        rounded = (
            round(raw_pred / self.config.rounding_increment) *
            self.config.rounding_increment
        )

        return max(0.5, rounded)  # Minimum 0.5

    def generate_lines(
        self,
        player_games: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic lines for all players/stats.

        Args:
            player_games: Player game logs
            target_date: Optional date to generate lines for
                        (defaults to latest date in data)

        Returns:
            DataFrame matching prop_lines schema:
                - game_id, game_date, player_id, stat, line, over_odds, under_odds
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        logger.info("Generating synthetic lines...")

        # Compute features
        player_games_features = compute_line_features(
            player_games,
            stat_cols=self.config.stat_cols,
            windows=self.config.rolling_windows
        )

        # Use latest date if not specified
        if target_date is None:
            target_date = player_games_features["game_date"].max()

        records = []

        # Get unique players
        players = player_games_features["player_id"].unique()

        for player_id in tqdm(players, desc="Generating lines", leave=False):
            player_df = player_games_features[
                player_games_features["player_id"] == player_id
            ].sort_values("game_date")

            if len(player_df) < 5:
                continue

            # Use latest row for features
            latest = player_df.iloc[-1]

            for stat in self.config.stat_cols:
                if stat not in player_df.columns:
                    continue

                # Extract features
                rolling_avgs = {}
                rolling_stds = {}

                for window in self.config.rolling_windows:
                    avg_col = f"{stat}_rolling_avg_{window}"
                    std_col = f"{stat}_rolling_std_{window}"

                    if avg_col in latest.index:
                        rolling_avgs[window] = latest[avg_col]
                    if std_col in latest.index:
                        rolling_stds[window] = latest[std_col]

                career_avg = latest.get(f"{stat}_career_avg", np.nan)
                career_var = latest.get(f"{stat}_career_var", np.nan)
                cv = latest.get(f"{stat}_cv", np.nan)
                games_played = latest.get("games_played", 0)

                # Skip if insufficient data
                if pd.isna(career_avg) or career_avg <= 0:
                    continue

                # Predict line
                try:
                    line = self.predict_line(
                        stat=stat,
                        rolling_avgs=rolling_avgs,
                        rolling_stds=rolling_stds,
                        career_avg=career_avg,
                        career_var=career_var,
                        cv=cv if not pd.isna(cv) else 0.0,
                        games_played=int(games_played),
                    )
                except Exception as e:
                    logger.warning(f"Error predicting line for {player_id}/{stat}: {e}")
                    continue

                # Generate odds from config range
                over_odds = round(
                    np.random.uniform(
                        self.config.default_odds_min,
                        self.config.default_odds_max
                    ),
                    2
                )
                under_odds = round(
                    np.random.uniform(
                        self.config.default_odds_min,
                        self.config.default_odds_max
                    ),
                    2
                )

                records.append({
                    "game_id": latest.get("game_id", f"gen_{player_id}_{stat}"),
                    "game_date": target_date,
                    "player_id": player_id,
                    "stat": stat,
                    "line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "book": "learned_generator",
                })

        result_df = pd.DataFrame(records)
        logger.info(f"Generated {len(result_df)} synthetic lines for {len(players)} players")

        return result_df

    def save(self, path: Path) -> None:
        """
        Save trained model to disk.

        Args:
            path: Path to save model (should end in .joblib)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self.model,
            "config": self.config,
            "metrics": self.metrics,
            "feature_cols": self.feature_cols,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(save_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "LearnedLineGenerator":
        """
        Load trained model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded LearnedLineGenerator instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_data = joblib.load(path)

        generator = cls(config=save_data["config"])
        generator.model = save_data["model"]
        generator.metrics = save_data["metrics"]
        generator.feature_cols = save_data["feature_cols"]
        generator.numeric_features = save_data["numeric_features"]
        generator.categorical_features = save_data["categorical_features"]
        generator.is_fitted = save_data["is_fitted"]

        logger.info(f"Model loaded from {path}")

        return generator


# Convenience functions

def train_line_generator(
    player_games: pd.DataFrame,
    real_lines: pd.DataFrame,
    config: Optional[LineGeneratorConfig] = None,
    verbose: bool = True
) -> Tuple[LearnedLineGenerator, LineGeneratorMetrics]:
    """
    Train a line generator model.

    Args:
        player_games: Player game logs
        real_lines: Real betting lines CSV data
        config: Optional configuration
        verbose: Log progress

    Returns:
        Tuple of (trained model, metrics)
    """
    generator = LearnedLineGenerator(config=config)
    metrics = generator.fit(player_games, real_lines, verbose=verbose)

    return generator, metrics


def generate_synthetic_lines(
    model: LearnedLineGenerator,
    player_games: pd.DataFrame,
    fallback_to_simple: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic lines using trained model.

    Falls back to simple rolling average method if model not fitted.

    Args:
        model: Trained LearnedLineGenerator
        player_games: Player game logs
        fallback_to_simple: Whether to fall back to simple method

    Returns:
        DataFrame with synthetic prop lines
    """
    if model.is_fitted:
        return model.generate_lines(player_games)
    elif fallback_to_simple:
        logger.warning("Model not fitted, falling back to simple method")
        from ..data_loader import _generate_synthetic_prop_lines
        return _generate_synthetic_prop_lines(player_games)
    else:
        raise RuntimeError("Model not fitted and fallback disabled")


def load_real_lines_csv(path: Path) -> pd.DataFrame:
    """
    Load and validate real betting lines CSV.

    Args:
        path: Path to CSV file

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If validation fails
    """
    is_valid, errors = validate_real_lines_csv(path)

    if not is_valid:
        raise ValueError(f"Invalid CSV format:\n" + "\n".join(errors))

    df = pd.read_csv(path, parse_dates=["game_date"])

    # Normalize stat names
    df["stat"] = df["stat"].str.lower().str.strip()

    # Ensure required columns
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    logger.info(f"Loaded {len(df)} real betting lines from {path}")
    logger.info(f"  Stats: {df['stat'].value_counts().to_dict()}")
    logger.info(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    return df


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to decimal odds.

    American odds examples:
        -110 -> 1.909 (favorite)
        +150 -> 2.50 (underdog)
        -200 -> 1.50 (heavy favorite)
        +100 -> 2.00 (even)

    Args:
        american_odds: American odds value (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 1.909, 2.50)
    """
    if pd.isna(american_odds):
        return np.nan

    try:
        odds = float(american_odds)
    except (ValueError, TypeError):
        return np.nan

    if odds >= 100:
        # Positive odds: underdog
        return 1 + (odds / 100)
    elif odds <= -100:
        # Negative odds: favorite
        return 1 + (100 / abs(odds))
    else:
        # Invalid range
        return np.nan


def transform_wide_lines_to_long(
    wide_df: pd.DataFrame,
    game_date: Optional[str] = None,
    player_id_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Transform wide-format betting lines to long format.

    Wide format (input):
        Team, Player, Points_Line, Points_Over_Odds, Points_Under_Odds,
        Rebounds_Line, Rebounds_Over_Odds, ...

    Long format (output):
        player_id, player_name, stat, line, over_odds, under_odds, game_date

    Args:
        wide_df: DataFrame in wide format
        game_date: Optional game date (defaults to today)
        player_id_map: Optional dict mapping player names to IDs

    Returns:
        DataFrame in long format for use with line generator
    """
    from datetime import date

    logger.info("Transforming wide-format lines to long format...")

    if game_date is None:
        game_date = str(date.today())

    records = []

    # Map column names to stats
    stat_mappings = {
        "pts": ("Points_Line", "Points_Over_Odds", "Points_Under_Odds"),
        "reb": ("Rebounds_Line", "Rebounds_Over_Odds", "Rebounds_Under_Odds"),
        "ast": ("Assists_Line", "Assists_Over_Odds", "Assists_Under_Odds"),
    }

    for _, row in wide_df.iterrows():
        player_name = row.get("Player", "")
        team = row.get("Team", "")

        # Generate player_id from name if not provided
        if player_id_map and player_name in player_id_map:
            player_id = player_id_map[player_name]
        else:
            # Create a simple hash-based ID from name
            player_id = str(abs(hash(player_name)) % 1000000)

        for stat, (line_col, over_col, under_col) in stat_mappings.items():
            # Get line value
            line = row.get(line_col)
            if pd.isna(line) or line == "":
                continue

            try:
                line = float(line)
            except (ValueError, TypeError):
                continue

            # Get odds and convert from American to decimal
            over_odds_raw = row.get(over_col, np.nan)
            under_odds_raw = row.get(under_col, np.nan)

            over_odds = american_to_decimal(over_odds_raw)
            under_odds = american_to_decimal(under_odds_raw)

            # If over odds missing, estimate from under odds (approximate)
            if pd.isna(over_odds) and not pd.isna(under_odds):
                # Rough estimate: if under is 1.9, over is ~1.9
                over_odds = round(2.0 - (under_odds - 1.9) * 0.5, 3)
            elif pd.isna(under_odds) and not pd.isna(over_odds):
                under_odds = round(2.0 - (over_odds - 1.9) * 0.5, 3)

            # Default to standard vig if both missing
            if pd.isna(over_odds):
                over_odds = 1.91
            if pd.isna(under_odds):
                under_odds = 1.91

            records.append({
                "player_id": player_id,
                "player_name": player_name,
                "team": team,
                "stat": stat,
                "line": line,
                "over_odds": round(over_odds, 3),
                "under_odds": round(under_odds, 3),
                "game_date": game_date,
                "book": "imported",
            })

    result_df = pd.DataFrame(records)

    logger.info(f"Transformed {len(result_df)} lines from {len(wide_df)} players")
    logger.info(f"  Stats: {result_df['stat'].value_counts().to_dict()}")

    return result_df


def load_wide_format_lines(
    path: Path,
    game_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and transform wide-format betting lines CSV.

    Handles American odds conversion and format transformation.

    Args:
        path: Path to wide-format CSV
        game_date: Optional game date

    Returns:
        DataFrame in long format ready for analysis
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading wide-format lines from {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows")

    # Transform to long format
    long_df = transform_wide_lines_to_long(df, game_date=game_date)

    return long_df
