"""
Feature extraction from NetworkX graphs.

All features are cheap to compute (O(n) or O(n+m) at most) so they
introduce negligible overhead compared to the algorithm being timed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def extract_features(G: nx.Graph) -> dict[str, float]:
    """Return a flat dict of numeric graph properties.

    Features
    --------
    num_nodes         : int   – number of nodes
    num_edges         : int   – number of edges
    density           : float – graph density (0–1)
    avg_degree        : float – mean degree
    max_degree        : int   – maximum degree
    min_degree        : int   – minimum degree
    std_degree        : float – standard deviation of degree sequence
    is_directed       : int   – 1 if directed, 0 otherwise
    num_components    : int   – number of (weakly) connected components
    self_loops        : int   – number of self-loops
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "density": 0.0,
            "avg_degree": 0.0, "max_degree": 0, "min_degree": 0,
            "std_degree": 0.0, "is_directed": int(G.is_directed()),
            "num_components": 0, "self_loops": 0,
        }

    degrees = np.array([d for _, d in G.degree()])
    avg_deg = float(degrees.mean())
    std_deg = float(degrees.std())
    max_deg = int(degrees.max())
    min_deg = int(degrees.min())

    if G.is_directed():
        n_comp = nx.number_weakly_connected_components(G)
    else:
        n_comp = nx.number_connected_components(G)

    return {
        "num_nodes": n,
        "num_edges": m,
        "density": nx.density(G),
        "avg_degree": avg_deg,
        "max_degree": max_deg,
        "min_degree": min_deg,
        "std_degree": std_deg,
        "is_directed": int(G.is_directed()),
        "num_components": n_comp,
        "self_loops": nx.number_of_selfloops(G),
    }


# Ordered list of feature names – used to build consistent feature vectors.
FEATURE_NAMES: list[str] = list(extract_features(nx.empty_graph(1)).keys())


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Convert a feature dict to a list in canonical order (FEATURE_NAMES)."""
    return [features[k] for k in FEATURE_NAMES]
