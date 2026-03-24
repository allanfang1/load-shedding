from collections import defaultdict
import math
import random
import time
import networkx as nx
from load_shed_manager import LoadShedManager
from core.timed_linkedlist import TimedLL
from core.sparsifiers import modified_spectral_sparsify

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

        self.ingest_buffer = []

        self.load_shed_manager = (
            LoadShedManager(predictor, feature_builder=self.build_predictor_features)
            if predictor else None
        )

        random.seed(42)
        self.start_time_system = time.perf_counter()
    
    def warmStart(self):
        self.runAlgo(self.graph)

    def build_predictor_features(self, graph: nx.Graph, remaining_time: float) -> dict[str, float]:
        """Build model input features expected by modelling_s-trained predictor."""
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        is_directed = int(graph.is_directed())
        
        return {
            "pre_num_nodes": float(n),
            "pre_num_edges": float(m),
            "pre_log_num_nodes": float(math.log2(n)) if n > 0 else 0.0,
            "pre_log_num_edges": float(math.log2(m)) if m > 0 else 0.0,
            "pre_is_directed": float(is_directed),
            "budget": max(0.0, float(remaining_time)),
        }

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

        print(f"Edge count: {self.edge_count}")
        print(f"Timed list size: {self.timed_list.size}")

        remaining_time = close_time + self.slide - time.perf_counter()

        # LOAD SHEDDING Derive shed parameter TODO validate this works
        if self.load_shed_manager is not None:
            predicted_s = self.load_shed_manager.predict(self.graph, remaining_time)
            shed_param = float(predicted_s)
            print(f"Load shed parameter: {shed_param} "
                  f"(remaining_time={remaining_time:.4f}s, "
                  f"current_edges={self.graph.number_of_edges()})")
            self.modifiedSpectralSparsity(close_time, s=shed_param)

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
                print(f"Edge added: {s} -> {d}, time: {t}")

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
        while self.timed_list.head and self.timed_list.head.t < t:
            s, d, edge_t = self.timed_list.popleft()
            self.removeEdge(s, d)
            print(f"Edge removed: {s} -> {d}, time: {edge_t}")


    def removeEdge(self, s, d):
        """Decrement multiplicity of edge (s, d) and remove from graph if count reaches 0.
        Returns True if edge was removed from graph, False otherwise."""
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
            return True
        return False

# ======================================================================
# Sparsifiers
# ======================================================================

    def modifiedSpectralSparsity(self, end_time, s: float):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        """
        For each edge in the current window, compute the probability p of keeping it based on the formula:
        P(keep) = davg * s / x
        where x = min(degAout, degBin)
        
        Intuition: keep all edges with degree < davg * s, hyperbolic decay proportional to 1/x"""
        edges_shed = modified_spectral_sparsify(
            timed_list=self.timed_list,
            graph=self.graph,
            remove_edge_fn=self.removeEdge,
            s=s,
            end_time=end_time,
        )

        print(f"Edges shed: {edges_shed}")


    def randomSparsity(self, s): # TODO
        pass