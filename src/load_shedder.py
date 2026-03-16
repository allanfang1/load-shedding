"""
Load shedder: derives how many edges to shed so that the predicted
algorithm runtime fits within the remaining wall-clock budget.

The module is *policy-agnostic* — it computes a target edge count
(and therefore the number of edges to drop) but does NOT decide
*which* edges to remove.  A sparsifier / shedding policy should be
plugged in separately.

Cost: O(log m) RF predictions.  No graph traversal — only O(1)
      calls to number_of_nodes() / number_of_edges().
"""

from __future__ import annotations

import networkx as nx

from modelling.feature_extraction import features_from_nm
from modelling.runtime_predictor import RuntimePredictor


class LoadShedder:
    """Derive the number of edges to shed given a time budget.

    Parameters
    ----------
    predictor : RuntimePredictor
        A trained ML model that predicts algorithm runtime from graph features.
    safety_margin : float
        Fraction of *remaining_time* to reserve as a safety buffer (0-1).
        E.g. 0.1 means the algorithm must fit in 90 % of the remaining time.
    """

    def __init__(self, predictor: RuntimePredictor, safety_margin: float = 0.1):
        self.predictor = predictor
        self.safety_margin = safety_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edges_to_shed(
        self,
        graph: nx.Graph,
        remaining_time: float,
    ) -> int:
        """Return the number of edges that must be shed.

        All graph queries are O(1).  The cost is O(log m) RF predictions,
        each of which is a scaler transform + tree traversal (microseconds).

        Parameters
        ----------
        graph : nx.Graph | nx.DiGraph
            Current snapshot of the windowed graph.
        remaining_time : float
            Wall-clock seconds left before the window deadline.

        Returns
        -------
        int
            Number of edges to shed.  0 means no shedding is needed.
        """
        n = graph.number_of_nodes()          # O(1)
        current_m = graph.number_of_edges()  # O(1)
        is_directed = int(graph.is_directed())

        if current_m == 0:
            return 0

        budget = remaining_time * (1.0 - self.safety_margin)
        if budget <= 0:
            return current_m

        # Fast path: predict with current n, m
        base_features = features_from_nm(n, current_m, is_directed)
        predicted = self.predictor.predict(base_features)
        if predicted <= budget:
            return 0  # graph already fits in budget

        # Binary search for the maximum edge count that fits in budget.
        # lo = 0 edges (trivially fits), hi = current edge count (does not fit)
        lo, hi = 0, current_m
        target_m = 0  # safe default: drop everything

        while lo <= hi:
            mid = (lo + hi) // 2
            est_features = features_from_nm(n, mid, is_directed)
            pred = self.predictor.predict(est_features)
            if pred <= budget:
                target_m = mid   # mid edges fit — try keeping more
                lo = mid + 1
            else:
                hi = mid - 1     # too slow — need fewer edges

        edges_to_drop = current_m - target_m
        return max(0, edges_to_drop)

