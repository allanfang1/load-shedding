from collections import deque, defaultdict
import math
import random
import time
import networkx as nx
import asyncio
# from sparsifiers import 
from buckets import Buckets
from timed_linkedlist import TimedLL, TimedLLNode
from load_shedder import LoadShedder

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

        # Load shedding
        self.load_shedder = LoadShedder(predictor) if predictor else None

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

        print(f"Bucket: {self.buckets.getCount(close_time - self.window_size)} edges in current bucket")
        print(f"Edge count: {self.edge_count}")
        print(f"Timed list size: {self.timed_list.size}")

        remaining_time = close_time + self.window_size - time.perf_counter()

        # Derive how many edges to shed
        edges_to_shed = 0
        if self.load_shedder is not None:
            edges_to_shed = self.load_shedder.edges_to_shed(self.graph, remaining_time)
            print(f"Load shedder: {edges_to_shed} edges to shed "
                  f"(remaining_time={remaining_time:.4f}s, "
                  f"current_edges={self.graph.number_of_edges()})")

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

    def getAverageDegree(self):
        return 2 * len(self.edge_count) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0

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
    
    def getEdgeCount(self):
        return sum(self.edge_count.values())