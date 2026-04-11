from collections import defaultdict
import networkx as nx
from core.timed_linkedlist import TimedLL, TimedLLNode
from core.sparsifiers import modified_spectral_sparsify
from core.moments import Moments

class MockWindowManager:
    """
    Single logical window:
    - keeps all added edges until you explicitly remove them
    - preserves edge arrival order via TimedLL
    - tracks duplicates via edge_count[(src, dst)]
    """

    def __init__(self, graph, algo=None, base_time=None):
        self.graph = graph
        self.algo = algo
        self.base_time = base_time  # optional guard; None means accept all t

        self.timed_list = TimedLL()
        self.edge_count = defaultdict(int)
        self.degree_count = defaultdict(int)
        self.in_moments = Moments()
        self.out_moments = Moments()

    def stageEdge(self, s, d, t):
        """Stage an edge into window structures without adding it to graph."""
        if self.base_time is not None and t < self.base_time:
            return
        self.timed_list.append(s, d, t)
        self.edge_count[(s, d)] += 1
        if self.edge_count[(s, d)] > 1:
            return

        self.degree_count[s] += 1
        self.degree_count[d] += 1
        self.in_moments.increment_update(self.degree_count[d]-1)
        self.out_moments.increment_update(self.degree_count[s]-1)

    def materialize_prefix(self, count: int) -> TimedLLNode | None:
        """Materialize first ``count`` staged arrivals into graph.

        Returns pointer to first non-materialized node, or None if all nodes were
        materialized.
        """
        remaining = max(0, int(count))
        curr = self.timed_list.head
        while curr is not None and remaining > 0:
            self.graph.add_edge(curr.src, curr.dst)
            curr = curr.next
            remaining -= 1
        return curr

    def removeBefore(self, t):
        while self.timed_list.head and self.timed_list.head.t < t:
            s, d, _ = self.timed_list.popleft()
            self._decrementEdge(s, d)

    def runAlgo(self, snapshot=None):
        if self.algo is None:
            return None
        return self.algo(self.graph if snapshot is None else snapshot)

    def earlyRemoveEdge(self, s, d):
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            self.in_moments.decrement_update(self.degree_count[d])
            self.out_moments.decrement_update(self.degree_count[s])
            self.degree_count[s] -= 1
            self.degree_count[d] -= 1
            if self.degree_count[s] == 0:
                del self.degree_count[s]
            if self.degree_count[d] == 0:
                del self.degree_count[d]
            del self.edge_count[(s, d)]
            return True
        return False

    def _decrementEdge(self, s, d):
        """Decrement multiplicity of edge (s, d) and remove from graph if count reaches 0.
        Returns True if edge was removed from graph, False otherwise."""
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            self.in_moments.decrement_update(self.degree_count[d])
            self.out_moments.decrement_update(self.degree_count[s])
            self.degree_count[s] -= 1
            self.degree_count[d] -= 1
            self.graph.remove_edge(s, d)
            if s in self.graph and self.graph.degree(s) == 0:
                self.graph.remove_node(s)
            if s in self.degree_count and self.degree_count[s] == 0:
                del self.degree_count[s]
            if d in self.graph and self.graph.degree(d) == 0:
                self.graph.remove_node(d)
            if d in self.degree_count and self.degree_count[d] == 0:
                del self.degree_count[d]
            del self.edge_count[(s, d)]
            return True
        return False

# ======================================================================
# Sparsifiers
# ======================================================================

    def modifiedSpectralSparsity(self, s: float, start_node: TimedLLNode | None = None):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        """
        For each edge in the current window, compute the probability p of keeping it based on the formula:
        P(keep) = davg * s / x
        where x = min(degAout, degBin)
        
        Intuition: keep all edges with degree < davg * s, hyperbolic decay proportional to 1/x"""
        if start_node is None:
            start_node = self.timed_list.head
        if start_node is None:
            return 0
        vertex_count = len(self.degree_count)
        davg = self.in_moments.get_mean(vertex_count) if vertex_count > 0 else 0.0
        modified_spectral_sparsify(
            timed_list=self.timed_list,
            start_node=start_node,
            davg=davg,
            graph=self.graph,
            remove_edge_fn=self.earlyRemoveEdge,
            degree_count=self.degree_count,
            s=s,
            end_time=None,
        )
    
    def randomSparsity(self, s): # TODO
        pass