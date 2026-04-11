from collections import defaultdict
import random
import networkx as nx
from core.timed_linkedlist import TimedLLNode, TimedLL


def modified_spectral_sparsify(
    timed_list: TimedLL,
    start_node: TimedLLNode,
    davg: float,
    remove_edge_fn,
    degree_count: defaultdict,
    graph: nx.Graph,
    s,
    end_time=None,
):
    """Apply modified spectral sparsification in-place.

    Parameters
    ----------
    timed_list : TimedLL
        Edge arrivals in linked-list order.
    davg : float
        Average degree of the graph.
    remove_edge_fn : Callable[[Any, Any], None]
        Callback that decrements multiplicity and removes graph edge/node when needed.
    s : float
        Sparsification strength parameter.
    end_time : float | None
        Optional upper timestamp bound (exclusive). If None, process full list.
    """

    curr = start_node

    removed = 0
    while curr and (end_time is None or curr.t < end_time):
        denom = min(degree_count[curr.src], degree_count[curr.dst])
        p = davg * s / denom if denom > 0 else 0

        temp = curr.next
        if p < 1 and random.random() >= p:
            timed_list.remove_node(curr)
            if remove_edge_fn(curr.src, curr.dst):
                removed += 1
        else:
            graph.add_edge(curr.src, curr.dst)
        curr = temp
    return removed
