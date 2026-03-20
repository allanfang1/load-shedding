from collections import deque, defaultdict
import math
import random
import time
import networkx as nx
import asyncio
from core.buckets import Buckets
from core.timed_linkedlist import TimedLL, TimedLLNode
from core.load_shed_manager import LoadShedManager

class WindowManager:
    def __init__(self, window_size, slide, graph, algo, base_time=0, predictor=None):
        """
        Parameters
        ----------
        window_size : int
            Size of the sliding window in **dataset timestamp units**.
        slide : int
            Slide interval in **dataset timestamp units**.
        base_time : int
            First expected timestamp in dataset units.
        predictor : RuntimePredictor | None
            Trained ML model for algorithm runtime prediction.
            When provided, enables automatic load shedding.
        """
        if slide > window_size:
            raise ValueError("slide should be less than or equal to window_size")
        self.window_size = window_size
        self.slide = slide
        self.base_time = base_time
        self.graph = graph
        self.algo = algo

        self.window_start = base_time
        
        self.timed_list = TimedLL()
        self.edge_count = defaultdict(int)
        self.window_log = "timing_log.txt"
        self.algo_log = "algo_log.txt"

        self.buckets = Buckets(self.base_time, self.slide)
        self.ingest_buffer = []

        self.load_shed_manager = LoadShedManager(predictor) if predictor else None

        random.seed(42)
        self.start_time_system = time.perf_counter()
    
    def warmStart(self):
        self.runAlgo(self.graph)

    def addEdge(self, s, d, t):
        if t < self.window_start:
            return
        print(f"Edge received: {s} -> {d}, time: {t}")
        self.ingest_buffer.append((s, d, t))
    
    def runCompleteWindow(self, close_time):
        # add remaining edges in buffer
        print(f"Running complete window at time {close_time - self.window_size} to {close_time} ({self.window_size})")
        self.batchAddEdges(self.ingest_buffer)
        self.ingest_buffer = []

        # shift window to close_time
        self.removeBefore(close_time - self.window_size)

        print(f"Bucket: {close_time - self.window_size} - {self.buckets.getCount(close_time - self.window_size)} edges in current bucket")
        print(f"Edge count: {self.edge_count}")
        print(f"Timed list size: {self.timed_list.size}")

        remaining_time = close_time + self.window_size - time.perf_counter()

        # Derive how many edges to shed
        edges_to_shed = 0
        if self.load_shed_manager is not None:
            edges_to_shed = self.load_shed_manager.edges_to_shed(self.graph, remaining_time)
            print(f"Load shed manager: {edges_to_shed} edges to shed "
                  f"(remaining_time={remaining_time:.4f}s, "
                  f"current_edges={self.graph.number_of_edges()})")
        
        if edges_to_shed <= self.buckets.getCount(close_time - self.window_size):
            print(f"Shedding {edges_to_shed} edges from current bucket")

        # TODO: apply sparsification / shedding policy to remove `edges_to_shed` edges
        # self.applyShedding(edges_to_shed)

        # run algo
        # self.runAlgo(self.graph)
        # print(f"Window moved to [{self.window_start}, {self.window_start + self.window_size})")

        self.window_start = close_time - self.window_size + self.slide

    def batchAddEdges(self, edges):
        for s, d, t in edges:
            self.timed_list.append(s, d, t)
            self.edge_count[(s, d)] += 1
            self.graph.add_edge(s, d)
            if self.edge_count[(s, d)] == 1:
                self.buckets.addEdge(t)
                print(f"Edge added: {s} -> {d}, time: {t}")

    def sparsify(self): # TODO implement different sparsification strategies
        self.graph
        self.timed_list
        # return davgSparsify(self.graph, self.graph, 1, self.getAverageDegree())
        return self.graph.copy()

    def shedBefore(self, t, ):
        self.buckets.removeBefore(t)
        while self.timed_list and self.timed_list.head.t < t:
            s, d, edge_t = self.timed_list.popleft()
            self.removeEdge(s, d)
            print(f"Edge removed: {s} -> {d}, time: {edge_t}")

    def runAlgo(self, snapshot):
        start_time = time.perf_counter()
        result = self.algo(snapshot)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        with open(self.algo_log, "a") as f:
            f.write(
                f"{start_time}, "
                f"{end_time}, "
                f"{elapsed},"
                f"{result}\n"
            )
        # TODO: store or print results - compare to ground truth
    
    def removeBefore(self, t):
        self.buckets.removeBefore(t)
        while self.timed_list and self.timed_list.head.t < t:
            s, d, edge_t = self.timed_list.popleft()
            self.removeEdge(s, d)
            print(f"Edge removed: {s} -> {d}, time: {edge_t}")
    
    def removeEdge(self, s, d):
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            print(f"Edge removed from graph: {s} -> {d}")
            self.graph.remove_edge(s, d)
            if self.graph.degree(s) == 0:
                self.graph.remove_node(s)
                print(f"Node removed: {s}")
            if self.graph.degree(d) == 0:
                self.graph.remove_node(d)
                print(f"Node removed: {d}")
            del self.edge_count[(s, d)]
    
    # def getEdgeCount(self):
    #     return sum(self.edge_count.values())

# ======================================================================
# Sparsifiers
# ======================================================================

    def modifiedSpectralSparsity(self, end_time, s: float):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        """
        For each edge in the current window, compute the probability p of keeping it based on the formula:
        P(keep) = davg * s / x
        where x = min(degAout, degBin)
        
        Intuition: keep all edges with degree < davg * s, hyperbolic decay proportional to 1/x"""
        davg = self.getAverageDegree(self.graph)
        curr = self.timed_list.head
        while curr and curr.t < end_time:
            denom = min(self.graph.out_degree(curr.src), self.graph.in_degree(curr.dst))
            p = davg * s / denom if denom > 0 else 0

            temp = curr.next
            if p >= 1 or random.random() >= p:
                self.timed_list.remove_node(curr) # remove edge from timed_list
                self.remove_edge(curr.src, curr.dst) ## TODO: this is not consistent with edge_count + graph - need to decrement edge_count and only remove from graph if count hits 0
            curr = temp

    def getAverageDegree(self, graph: nx.Graph) -> float:
            return 2 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0

    def randomSparsity(self, s): # TODO
        pass