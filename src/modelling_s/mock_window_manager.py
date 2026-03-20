from collections import defaultdict
import random
import networkx as nx
from core.timed_linkedlist import TimedLL

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

    # def removeOldestMatchingEdge(self, s, d):
    #     """
    #     Remove one occurrence of (s, d), specifically the oldest one in TimedLL.
    #     Keeps TimedLL + edge_count + graph consistent.
    #     Returns True if removed, False if not found.
    #     """
    #     curr = self.timed_list.head
    #     while curr:
    #         if curr.src == s and curr.dst == d:
    #             self.timed_list.remove_node(curr)
    #             self._decrementEdge(s, d)
    #             return True
    #         curr = curr.next
    #     return False

    def runAlgo(self, snapshot=None):
        if self.algo is None:
            return None
        return self.algo(self.graph if snapshot is None else snapshot)

    def _decrementEdge(self, s, d):
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            if self.graph.has_edge(s, d):
                self.graph.remove_edge(s, d)
            if s in self.graph and self.graph.degree(s) == 0:
                self.graph.remove_node(s)
            if d in self.graph and self.graph.degree(d) == 0:
                self.graph.remove_node(d)
            del self.edge_count[(s, d)]

# ======================================================================
# Sparsifiers
# ======================================================================

    def modifiedSpectralSparsity(self, s: float):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        """
        For each edge in the current window, compute the probability p of keeping it based on the formula:
        P(keep) = davg * s / x
        where x = min(degAout, degBin)
        
        Intuition: keep all edges with degree < davg * s, hyperbolic decay proportional to 1/x"""
        davg = self.getAverageDegree(self.graph)
        curr = self.timed_list.head
        while curr:
            denom = min(self.graph.out_degree(curr.src), self.graph.in_degree(curr.dst))
            p = davg * s / denom if denom > 0 else 0

            temp = curr.next
            if p < 1 and random.random() >= p:
                self.timed_list.remove_node(curr) # remove edge from timed_list
                self._decrementEdge(curr.src, curr.dst)
            curr = temp

    def getAverageDegree(self, graph: nx.Graph) -> float:
            return 2 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
    
    def randomSparsity(self, s): # TODO
        pass