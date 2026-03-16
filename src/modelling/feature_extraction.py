"""
Feature extraction from NetworkX graphs.

All features are O(1) — derived purely from node count, edge count,
and directedness.  This keeps both training and online inference
aligned and avoids any graph traversal.
"""

from __future__ import annotations

import networkx as nx


def extract_features(G: nx.Graph) -> dict[str, float]:
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

    return features_from_nm(n, m, is_dir)


def features_from_nm(n: int, m: int, is_directed: int = 1) -> dict[str, float]:
    """Build the feature dict from just node/edge counts.

    Usable both during training (from a real graph's n, m) and during
    online inference (from O(1) graph queries), guaranteeing the model
    sees the same feature distribution in both cases.
    """
    if n == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "density": 0.0,
            "avg_degree": 0.0, "is_directed": is_directed,
        }

    denom = n * (n - 1) if is_directed else n * (n - 1) / 2
    density = m / denom if denom else 0.0
    avg_degree = (m / n) if is_directed else (2 * m / n)

    return {
        "num_nodes": n,
        "num_edges": m,
        "density": density,
        "avg_degree": avg_degree,
        "is_directed": is_directed,
    }


# Ordered list of feature names - used to build consistent feature vectors.
FEATURE_NAMES: list[str] = list(features_from_nm(0, 0).keys())


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Convert a feature dict to a list in canonical order (FEATURE_NAMES)."""
    return [features[k] for k in FEATURE_NAMES]