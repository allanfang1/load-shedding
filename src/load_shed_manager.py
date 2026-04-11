"""Prediction interface around ``RuntimePredictor``.

This module builds input features and runs runtime prediction.
"""

from __future__ import annotations

from typing import Any, Callable

from modelling_s.runtime_predictor import RuntimePredictor
from core.moments import Moments


class LoadShedManager:
    """Thin interface for runtime prediction.

    Parameters
    ----------
    predictor : RuntimePredictor
        Trained model used for inference.
    feature_builder : Callable
        Function that builds the feature dict passed to ``predictor.predict``.
        Signature is flexible and determined by the caller.
    """

    def __init__(
        self,
        predictor: RuntimePredictor,
        feature_builder: Callable[..., dict[str, float]],
    ):
        self.predictor = predictor
        self.feature_builder = feature_builder

    def predict(self, *feature_builder_args: Any) -> float:
        """Build features and run model inference.

        Accepts passthrough positional arguments so callers can choose the
        feature-builder shape (e.g., graph-based or precomputed stats).
        """
        features = self.feature_builder(*feature_builder_args)
        model_features = list(getattr(self.predictor, "feature_names", []))
        if model_features and list(features.keys()) != model_features:
            raise ValueError(
                "feature_builder output keys do not match model feature schema from meta.json. "
                f"Model: {model_features}, Built: {list(features.keys())}"
            )
        return float(self.predictor.predict(features))