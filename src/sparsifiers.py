import networkx as nx

def davgSparsify(graph, s, davg): # for a -> bdavg * s / min(degAout, degBin)
    sparsified = nx.DiGraph()
    for edge in snapshot.edges():
        u, v = edge
        if snapshot.degree(u) > 0 and snapshot.degree(v) > 0:
            if snapshot.degree(u) < davg * s / snapshot.degree(v):
                sparsified.add_edge(u, v)
    return sparsified