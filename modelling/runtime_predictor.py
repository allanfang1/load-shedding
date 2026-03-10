"""
ML-based runtime predictor.

Uses a Random Forest regressor (fast inference, handles non-linear
relationships, needs minimal tuning).  The trained model + scaler are
persisted with joblib so prediction is a single cheap call.
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from feature_extraction import FEATURE_NAMES, features_to_vector


class RuntimePredictor:
    """Train once, predict many times."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,          # use all cores for training
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.algorithm_name: str | None = None
        self.cv_scores: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithm_name: str = "unknown",
        cv_folds: int = 5,
    ) -> dict:
        """Fit the model and return evaluation metrics.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) – runtimes in seconds
        algorithm_name : stored for later reference
        cv_folds : number of cross-validation folds (0 to skip)

        Returns
        -------
        dict with keys: mae, r2, cv_mean, cv_std  (cv_* only if cv_folds > 0)
        """
        self.algorithm_name = algorithm_name
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        y_pred = self.model.predict(X_scaled)
        metrics: dict = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        }

        if cv_folds > 0 and len(y) >= cv_folds:
            self.cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv_folds, scoring="neg_mean_absolute_error",
            )
            metrics["cv_mean_mae"] = float(-self.cv_scores.mean())
            metrics["cv_std_mae"] = float(self.cv_scores.std())

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: dict[str, float]) -> float:
        """Predict runtime (seconds) from a single feature dict.

        This is the hot-path call in a streaming system – it's just
        a scaler transform + tree traversal, i.e. microseconds.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet.  Call fit() first.")
        vec = np.array(features_to_vector(features)).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        return float(self.model.predict(vec_scaled)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict runtimes for multiple feature vectors at once."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """Return feature name → importance (Gini-based)."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return dict(zip(FEATURE_NAMES, self.model.feature_importances_))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save model, scaler, and metadata to *directory*."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.joblib"))
        meta = {
            "algorithm_name": self.algorithm_name,
            "feature_names": FEATURE_NAMES,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "RuntimePredictor":
        """Load a previously saved predictor."""
        predictor = cls()
        predictor.model = joblib.load(os.path.join(directory, "model.joblib"))
        predictor.scaler = joblib.load(os.path.join(directory, "scaler.joblib"))
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        predictor.algorithm_name = meta.get("algorithm_name")
        predictor._is_fitted = True
        return predictor
