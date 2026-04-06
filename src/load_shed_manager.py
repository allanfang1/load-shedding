"""Prediction interface around ``RuntimePredictor``.

This module 1) builds input features, 2) calls the model, and 
3) optionally maps the raw prediction to a caller-defined output shape.
"""

from __future__ import annotations

import networkx as nx
from typing import Any, Callable

from modelling_s.runtime_predictor import RuntimePredictor
from core.moments import Moments


class LoadShedManager:
    """Thin, configurable interface for runtime prediction.

    Parameters
    ----------
    predictor : RuntimePredictor
        Trained model used for inference.
    feature_builder : Callable
        Function that builds the feature dict passed to ``predictor.predict``.
        Signature: ``(graph, remaining_time) -> dict[str, float]``.
    output_adapter : Callable | None
        Function that maps raw model output to any shape needed by callers.
        Signature: ``(prediction, features, graph, remaining_time) -> Any``.
        If ``None``, raw ``float`` prediction is returned.
    """

    def __init__(
        self,
        predictor: RuntimePredictor,
        feature_builder: Callable[[nx.Graph, float, Moments, Moments], dict[str, float]],
        output_adapter: Callable[[float, dict[str, float], nx.Graph, float], Any] | None = None,
    ):
        self.predictor = predictor
        self.feature_builder = feature_builder
        self.output_adapter = output_adapter

    def predict(
        self,
        graph: nx.Graph,
        remaining_time: float,
        in_moments: Moments,
        out_moments: Moments 
    ) -> Any:
        """Build features, run model inference, and adapt output if requested."""
        features = self.feature_builder(graph, remaining_time, in_moments, out_moments)
        model_features = list(getattr(self.predictor, "feature_names", []))
        if model_features and list(features.keys()) != model_features:
            raise ValueError(
                "feature_builder output keys do not match model feature schema from meta.json. "
                f"Model: {model_features}, Built: {list(features.keys())}"
            )
        prediction = self.predictor.predict(features)

        if self.output_adapter is None:
            return prediction
        return self.output_adapter(prediction, features, graph, remaining_time)