"""
Modular algorithm registry.

To add a new algorithm:
    1. Write a function that takes a NetworkX graph and returns a result.
    2. Register it in ALGORITHM_REGISTRY below.

Each entry maps a human-readable name to a callable: graph → result.
The callable is timed externally; it should NOT include I/O or printing.
"""

import networkx as nx


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------

def betweenness_centrality(G: nx.Graph) -> dict:
    """Compute betweenness centrality for all nodes."""
    return nx.betweenness_centrality(G)


def approx_betweenness_centrality(G: nx.Graph, k: int = 100) -> dict:
    """Approximate betweenness centrality (samples k pivot nodes).
    Much faster on large graphs: O(k*(n+m)) instead of O(n*m)."""
    k = min(k, G.number_of_nodes())
    return nx.betweenness_centrality(G, k=k)


def pagerank(G: nx.DiGraph) -> dict:
    """Compute PageRank.  Works on DiGraph; for undirected graphs the caller
    should convert first (see _ensure_directed helper in data_collector)."""
    return nx.pagerank(G)


def closeness_centrality(G: nx.Graph) -> dict:
    """Compute closeness centrality for all nodes."""
    return nx.closeness_centrality(G)


def clustering_coefficient(G: nx.Graph) -> dict:
    """Compute the clustering coefficient for each node."""
    return nx.clustering(G)


# ---------------------------------------------------------------------------
# Registry  –  edit this dict to enable / disable algorithms
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: dict[str, callable] = {
    "betweenness_centrality": betweenness_centrality,
    "approx_betweenness_centrality": approx_betweenness_centrality,
    "pagerank": pagerank,
    # Uncomment to enable:
    # "closeness_centrality": closeness_centrality,
    # "clustering_coefficient": clustering_coefficient,
}


def list_algorithms() -> list[str]:
    """Return the names of all registered algorithms."""
    return list(ALGORITHM_REGISTRY.keys())


def get_algorithm(name: str) -> callable:
    """Look up an algorithm by name.  Raises KeyError if not found."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list_algorithms()}"
        )
    return ALGORITHM_REGISTRY[name]