from collections import defaultdict
import networkx as nx
from core.timed_linkedlist import TimedLL
from core.sparsifiers import modified_spectral_sparsify

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

    def addEdge(self, s, d, t):
        if self.base_time is not None and t < self.base_time:
            return
        self.timed_list.append(s, d, t)
        self.edge_count[(s, d)] += 1
        self.graph.add_edge(s, d)

    def removeBefore(self, t):
        while self.timed_list.head and self.timed_list.head.t < t:
            s, d, _ = self.timed_list.popleft()
            self._decrementEdge(s, d)

    def runAlgo(self, snapshot=None):
        if self.algo is None:
            return None
        return self.algo(self.graph if snapshot is None else snapshot)

    def _decrementEdge(self, s, d):
        """Decrement multiplicity of edge (s, d) and remove from graph if count reaches 0.
        Returns True if edge was removed from graph, False otherwise."""
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            if self.graph.has_edge(s, d):
                self.graph.remove_edge(s, d)
            if s in self.graph and self.graph.degree(s) == 0:
                self.graph.remove_node(s)
            if d in self.graph and self.graph.degree(d) == 0:
                self.graph.remove_node(d)
            del self.edge_count[(s, d)]
            return True
        return False

# ======================================================================
# Sparsifiers
# ======================================================================

    def modifiedSpectralSparsity(self, s: float):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        """
        For each edge in the current window, compute the probability p of keeping it based on the formula:
        P(keep) = davg * s / x
        where x = min(degAout, degBin)
        
        Intuition: keep all edges with degree < davg * s, hyperbolic decay proportional to 1/x"""
        modified_spectral_sparsify(
            timed_list=self.timed_list,
            graph=self.graph,
            remove_edge_fn=self._decrementEdge,
            s=s,
            end_time=None,
        )
    
    def randomSparsity(self, s): # TODO
        pass