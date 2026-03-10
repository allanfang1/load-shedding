import networkx as nx

def getAverageDegree(graph: nx.Graph, edge_count = None) -> float:
        if edge_count is None:
            edge_count = graph.number_of_edges()
        return 2 * edge_count / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0

def getEdgeCountWithDuplicates(edge_count: dict) -> int:
        return sum(edge_count.values())