"""
ML-based runtime predictor.

Uses a Random Forest regressor (fast inference, handles non-linear
relationships, needs minimal tuning).  The trained model is
persisted with joblib so prediction is a single cheap call.
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from modelling_s_v2.feature_extraction import FEATURE_NAMES
except ModuleNotFoundError:
    from feature_extraction import FEATURE_NAMES


class RuntimePredictor:
    """Train once, predict many times."""

    def __init__(
        self,
        n_estimators: int = 120,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            oob_score=True,
            random_state=random_state,
            n_jobs=-1,          # use all cores for training
        )
        self._is_fitted = False
        self.algorithm_name: str | None = None
        self.cv_scores: np.ndarray | None = None
        self.feature_names: list[str] = list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithm_name: str = "unknown",
        cv_folds: int = 0,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """Fit the model and return evaluation metrics.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) - runtimes in seconds
        algorithm_name : stored for later reference
        cv_folds : number of cross-validation folds (0 to skip)
        sample_weight : optional per-sample weights for fitting

        Returns
        -------
        dict with keys: mae, r2, cv_mean, cv_std  (cv_* only if cv_folds > 0)
        """
        self.algorithm_name = algorithm_name
        if feature_names is not None:
            self.feature_names = list(feature_names)
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        self._is_fitted = True

        y_pred = self.model.predict(X)
        metrics: dict = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        }
        if hasattr(self.model, "oob_score_"):
            metrics["oob_r2"] = float(self.model.oob_score_)

        if cv_folds > 0 and len(y) >= cv_folds:
            n_splits = min(cv_folds, len(y))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            self.cv_scores = cross_val_score(
                self.model,
                X,
                y,
                cv=splitter,
                scoring="neg_mean_absolute_error",
            )
            metrics["cv_mean_mae"] = float(-self.cv_scores.mean())
            metrics["cv_std_mae"] = float(self.cv_scores.std())
            metrics["cv_mae_scores"] = [float(-s) for s in self.cv_scores]
            metrics["cv_folds_used"] = int(len(self.cv_scores))

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: dict[str, float]) -> float:
        """Predict runtime (seconds) from a single feature dict.

        This is the hot-path call in a streaming system - it's just
        a tree traversal, i.e. microseconds.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet.  Call fit() first.")
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(
                f"Missing feature(s) for prediction: {missing}. "
                f"Model expects: {self.feature_names}"
            )
        vec = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        return float(self.model.predict(vec)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict runtimes for multiple feature vectors at once."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """Return feature name → importance (Gini-based)."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return dict(zip(self.feature_names, self.model.feature_importances_))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save model and metadata to *directory*."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "model.joblib"))
        meta = {
            "algorithm_name": self.algorithm_name,
            "feature_names": self.feature_names,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "RuntimePredictor":
        """Load a previously saved predictor."""
        predictor = cls()
        predictor.model = joblib.load(os.path.join(directory, "model.joblib"))
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        predictor.algorithm_name = meta.get("algorithm_name")
        predictor.feature_names = meta.get("feature_names", list(FEATURE_NAMES))
        predictor._is_fitted = True
        return predictor