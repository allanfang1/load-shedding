from collections import defaultdict
import json
import math
import os
import csv
import random
import time
from queue import SimpleQueue, Empty
import networkx as nx
from load_shed_manager import LoadShedManager
from core.timed_linkedlist import TimedLL, TimedLLNode
from core.sparsifiers import modified_spectral_sparsify
from core.moments import Moments

class WindowManager:
    def __init__(self, window_size, slide, graph, algo, k = 10, base_time=0, predictor=None, headroom_percent=0.0):
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
        self.k = k
        self.headroom_seconds = (headroom_percent / 100.0) * self.slide

        self.window_start = base_time
        
        self.timed_list = TimedLL()
        self.edge_count = defaultdict(int)
        self.degree_count = defaultdict(int)
        self.in_moments = Moments()
        self.out_moments = Moments()

        self.ingest_queue = SimpleQueue()

        self.load_shed_manager = (
            LoadShedManager(predictor, feature_builder=self.build_predictor_features)
            if predictor else None
        )

        random.seed(42)
        self.start_time_system = time.perf_counter()
    
    def warmStart(self):
        self.runAlgo(self.graph)

    def build_predictor_features(self, edge_count: int, vertex_count: int, percent_incoming: float, remaining_time: float, in_moments: Moments, out_moments: Moments) -> dict[str, float]:
        """Build model input features expected by modelling_s-trained predictor."""
        n = vertex_count
        m = edge_count

        return { # TODO we got new features
            "pre_num_nodes": float(n),
            "pre_num_edges": float(m),
            "pre_log_num_nodes": float(math.log2(n)) if n > 0 else 0.0,
            "pre_log_num_edges": float(math.log2(m)) if m > 0 else 0.0,
            "percent_incoming": percent_incoming,
            "budget": max(0.0, float(remaining_time)),
            "pre_avg": in_moments.get_mean(n), # avg degree is same for in and out in directed graph
            "pre_var_in": in_moments.get_variance(n),
            "pre_var_out": out_moments.get_variance (n),
            "pre_skew_in": in_moments.get_skewness(n),
            "pre_skew_out": out_moments.get_skewness(n)
        }

    def addEdge(self, raw):
        parts = raw.split()
        ts = time.perf_counter()
        self.ingest_queue.put((int(parts[0]), int(parts[1]), ts))

    def _drain_ingest_queue(self):
        edges = []
        while True:
            try:
                edges.append(self.ingest_queue.get_nowait())
            except Empty:
                break
        return edges
    
    def runCompleteWindow(self, close_time):
        # add remaining edges in queue
        drained_edges = self._drain_ingest_queue()
        print(f"Running complete window at time {close_time - self.window_size} to {close_time} ({len(drained_edges)} buffered edges)")
        
        # shift window to close_time
        self.removeBefore(close_time - self.window_size)

        remaining_time = close_time + self.slide - time.perf_counter() - self.headroom_seconds
        shed_count = 0
        begin_use_budget = time.perf_counter()

        if self.load_shed_manager is not None:
            first_new_node, incoming_edge_count = self.batchAddEdgesShed(drained_edges)
            predicted_s = self.load_shed_manager.predict(
                len(self.edge_count),
                len(self.degree_count),
                (incoming_edge_count / len(self.edge_count)) if self.edge_count else 0.0,
                remaining_time, 
                self.in_moments, 
                self.out_moments)
            shed_param = float(predicted_s)
            shed_count = self.modifiedSpectralSparsity(close_time, first_new_node, s=shed_param)
        else:
            self.batchAddEdges(drained_edges)

        full_edge_count = self.graph.number_of_edges()

        result = self.runAlgo(self.graph)
        
        old_start = self.window_start
        self.window_start = close_time - self.window_size + self.slide

        return {"system_type": "shed" if self.load_shed_manager else "classic",
                         "window": old_start,
                         "window_size": self.window_size,
                         "slide": self.slide,
                         "incoming_edges": len(drained_edges),
                         "edge_count": full_edge_count,
                         "shed_count": shed_count,
                         "end_time": time.perf_counter(),
                         "budget": remaining_time,
                         "actual_runtime": time.perf_counter() - begin_use_budget,
                         f"pagerank_top{self.k}": json.dumps(result)
                         }
      

    def batchAddEdges(self, edges):
        for s, d, t in edges:
            self.timed_list.append(s, d, t)
            self.edge_count[(s, d)] += 1
            if self.edge_count[(s, d)] == 1:
                self.graph.add_edge(s, d)
                self.degree_count[s] += 1
                self.degree_count[d] += 1
                self.in_moments.increment_update(self.graph.in_degree(d)-1)
                self.out_moments.increment_update(self.graph.out_degree(s)-1)
    
    def batchAddEdgesShed(self, edges):
        first_new_node = None
        count = 0
        for s, d, t in edges:
            self.timed_list.append(s, d, t)
            if first_new_node is None:
                first_new_node = self.timed_list.tail
            self.edge_count[(s, d)] += 1
            if self.edge_count[(s, d)] == 1:
                count += 1
                self.degree_count[s] += 1
                self.degree_count[d] += 1
                self.in_moments.increment_update(self.graph.in_degree(d)-1)
                self.out_moments.increment_update(self.graph.out_degree(s)-1)
        return first_new_node, count

    def runAlgo(self, snapshot):
        result = self.algo(snapshot)

        return sorted(
            ((str(node), float(v)) for node, v in result.items()),
            key=lambda x: x[1],
            reverse=True
        )[:self.k]
        
    def removeBefore(self, t):
        while self.timed_list.head and self.timed_list.head.t < t:
            s, d, edge_t = self.timed_list.popleft()
            self.removeEdge(s, d)
            # print(f"Edge removed: {s} -> {d}, time: {edge_t}")

    def earlyRemoveEdge(self, s, d):
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            self.in_moments.decrement_update(self.graph.in_degree(d))
            self.out_moments.decrement_update(self.graph.out_degree(s))
            self.degree_count[s] -= 1
            self.degree_count[d] -= 1
            if self.degree_count[s] == 0:
                del self.degree_count[s]
            if self.degree_count[d] == 0:
                del self.degree_count[d]
            del self.edge_count[(s, d)]
            return True
        return False

    def removeEdge(self, s, d):
        """Decrement multiplicity of edge (s, d) and remove from graph if count reaches 0.
        Returns True if edge was removed from graph, False otherwise."""
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            # print(f"Edge removed from graph: {s} -> {d}")
            self.in_moments.decrement_update(self.graph.in_degree(d))
            self.out_moments.decrement_update(self.graph.out_degree(s))
            self.degree_count[s] -= 1
            self.degree_count[d] -= 1
            self.graph.remove_edge(s, d)
            if self.graph.degree(s) == 0:
                self.graph.remove_node(s)
                del self.degree_count[s]
            if self.graph.degree(d) == 0:
                self.graph.remove_node(d)
                del self.degree_count[d]
            del self.edge_count[(s, d)]
            return True
        return False

# ======================================================================
# Sparsifiers
# ======================================================================

    def modifiedSpectralSparsity(self, end_time, start_node: TimedLLNode, s: float):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        """
        For each edge in the current window, compute the probability p of keeping it based on the formula:
        P(keep) = davg * s / x
        where x = min(degAout, degBin)
        
        Intuition: keep all edges with degree < davg * s, hyperbolic decay proportional to 1/x"""
        edges_shed = modified_spectral_sparsify(
            timed_list=self.timed_list,
            start_node=start_node,
            graph=self.graph,
            davg=self.in_moments.get_mean(len(self.degree_count)),
            remove_edge_fn=self.earlyRemoveEdge,
            degree_count=self.degree_count,
            s=s,
            end_time=end_time,
        )

        return edges_shed
        # print(f"Edges shed: {edges_shed}")


    def randomSparsity(self, s): # TODO
        pass