import random


def modified_spectral_sparsify(
    timed_list,
    graph,
    remove_edge_fn,
    s,
    end_time=None,
):
    """Apply modified spectral sparsification in-place.

    Parameters
    ----------
    timed_list : TimedLL
        Edge arrivals in linked-list order.
    graph : nx.DiGraph | nx.Graph
        Current snapshot graph.
    remove_edge_fn : Callable[[Any, Any], None]
        Callback that decrements multiplicity and removes graph edge/node when needed.
    get_average_degree_fn : Callable[[graph], float]
        Graph average degree function.
    s : float
        Sparsification strength parameter.
    end_time : float | None
        Optional upper timestamp bound (exclusive). If None, process full list.
    """
    if graph.number_of_nodes() > 0:
        if graph.is_directed():
            davg = graph.number_of_edges() / graph.number_of_nodes()
        else:
            davg = 2 * graph.number_of_edges() / graph.number_of_nodes()
    else:
        davg = 0
        
    curr = timed_list.head

    removed = 0
    while curr and (end_time is None or curr.t < end_time):
        denom = min(graph.out_degree(curr.src), graph.in_degree(curr.dst))
        p = davg * s / denom if denom > 0 else 0

        temp = curr.next
        if p < 1 and random.random() >= p:
            timed_list.remove_node(curr)
            if remove_edge_fn(curr.src, curr.dst):
                removed += 1
        curr = temp
    return removed
