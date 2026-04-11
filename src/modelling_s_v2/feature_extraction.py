"""
Feature extraction from NetworkX graphs.

All features are O(1) — derived purely from node count, edge count,
and directedness.  This keeps both training and online inference
aligned and avoids any graph traversal.
"""

from __future__ import annotations

import networkx as nx
import math


def extract_features(G: nx.Graph, avg_in, avg_out, var_in, var_out, skew_in, skew_out) -> dict[str, float]:
    """Return a flat dict of numeric graph properties.

    Only uses O(1) calls: number_of_nodes(), number_of_edges(),
    is_directed().  All other features are derived arithmetically.

    Features
    --------
    num_nodes   : int   - number of nodes
    num_edges   : int   - number of edges
    density     : float - graph density (0-1)
    avg_degree  : float - mean degree  (m/n for directed, 2m/n for undirected)
    is_directed : int   - 1 if directed, 0 otherwise
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    is_dir = int(G.is_directed())

    return features_from_nm(n, m, is_dir, avg_in, avg_out, var_in, var_out, skew_in, skew_out)


def features_from_nm(n: int, m: int, is_directed: int = 1, avg_in: float = 0.0, avg_out: float = 0.0, var_in: float = 0.0, var_out: float = 0.0, skew_in: float = 0.0, skew_out: float = 0.0) -> dict[str, float]:
    """Build the feature dict from just node/edge counts.

    Usable both during training (from a real graph's n, m) and during
    online inference (from O(1) graph queries), guaranteeing the model
    sees the same feature distribution in both cases.
    """
    if n == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "log_num_nodes": 0.0, "log_num_edges": 0.0,
            "is_directed": is_directed, "avg_in": avg_in, "avg_out": avg_out, "var_in": var_in, "var_out": var_out, "skew_in": skew_in, "skew_out": skew_out
        }

    return {
        "num_nodes": n,
        "num_edges": m,
        "log_num_nodes": math.log2(n) if n > 0 else 0.0,
        "log_num_edges": math.log2(m) if m > 0 else 0.0, 
        "is_directed": is_directed,
        "avg_in": avg_in,
        "avg_out": avg_out,
        "var_in": var_in,
        "var_out": var_out,
        "skew_in": skew_in,
        "skew_out": skew_out
    }


# Ordered list of feature names - used to build consistent feature vectors.
FEATURE_NAMES: list[str] = list(features_from_nm(0, 0).keys())


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Convert a feature dict to a list in canonical order (FEATURE_NAMES)."""
    return [features[k] for k in FEATURE_NAMES]