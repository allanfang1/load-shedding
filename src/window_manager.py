from collections import defaultdict
import math
import random
import time
import networkx as nx
from buckets import Buckets
from timed_linkedlist import TimedLL
from helper import getAverageDegree
from system_manager import SystemManager, TimeCard


class WindowManager:
    def __init__(self, window_size, slide, graph, algo, base_time=0,
                 slide_budget=1.0, runtime_predictor=None):
        """
        Parameters
        ----------
        window_size : int
            Size of the sliding window in dataset timestamp units.
        slide : int
            Slide interval in dataset timestamp units.
        graph : nx.DiGraph
            The mutable graph maintained by this manager.
        algo : callable
            Graph algorithm executed each window (e.g. ``nx.pagerank``).
        base_time : int
            First expected timestamp in dataset units.
        slide_budget : float
            Wall-clock seconds allowed per processing cycle.
        runtime_predictor : object, optional
            ML predictor for algorithm runtime (passed to SystemManager).
        """
        if slide > window_size:
            raise ValueError("slide must be <= window_size")

        self.window_size = window_size
        self.slide = slide
        self.base_time = base_time
        self.graph = graph
        self.algo = algo

        self.window_start = base_time
        self.window_end = base_time + window_size

        self.timed_list = TimedLL()
        self.edge_count = defaultdict(int)
        self.window_log = "timing_log.txt"
        self.algo_log = "algo_log.txt"

        self.buckets = Buckets(self.base_time, self.slide)
        self.ingest_buffer = []

        self.sm = SystemManager(
            slide_budget,
            runtime_predictor=runtime_predictor,
        )

        random.seed(42)

    # ------------------------------------------------------------------
    # Edge processing
    # ------------------------------------------------------------------

    def addEdge(self, s, d, t):
        """Add edge (*s*, *d*) with timestamp *t*.

        When *t* falls outside the current window the full processing
        cycle fires:  expire → ingest → shed → algo → update stats.
        """
        if t < self.window_start:
            return

        if t >= self.window_end:
            cycle_start = time.perf_counter()

            # 1. EXPIRE — remove edges that fell out of the window
            self.sm.expiry_card.start = time.perf_counter()
            expired = self.shiftWindow(t)
            self.sm.expiry_card.end = time.perf_counter()
            self.sm.expiry_card.count = expired

            self.buckets.removeBefore(self.window_start)

            # 2. INGEST — flush the edge buffer into the graph
            self.sm.ingest_card.start = time.perf_counter()
            self.batchAddEdges(self.ingest_buffer)
            self.sm.ingest_card.end = time.perf_counter()
            self.sm.ingest_card.count = len(self.ingest_buffer)
            self.ingest_buffer = []

            # 3. SHED — drop edges if algo won't fit in remaining budget
            n_edges = len(self.edge_count)
            graph_features = None
            if self.sm.runtime_predictor is not None:
                try:
                    from feature_extraction import extract_features
                    graph_features = extract_features(self.graph)
                except ImportError:
                    pass

            shed_count = self.sm.compute_shed_count(
                n_edges,
                self.sm.expiry_card.elapsed,
                self.sm.ingest_card.elapsed,
                graph_features=graph_features,
            )

            self.sm.shed_card.start = time.perf_counter()
            if shed_count > 0:
                self.randomShed(shed_count)
            self.sm.shed_card.end = time.perf_counter()
            self.sm.shed_card.count = shed_count

            # 4. RUN ALGORITHM
            n_edges_after = self.graph.number_of_edges()
            self.sm.algo_card.start = time.perf_counter()
            self.runAlgo(self.graph)
            self.sm.algo_card.end = time.perf_counter()
            self.sm.algo_card.count = max(n_edges_after, 1)

            # 5. UPDATE running statistics & log
            self.sm.update_cycle_stats()

            cycle_end = time.perf_counter()
            with open(self.window_log, "a") as f:
                f.write(f"{cycle_start},{cycle_end},{cycle_end - cycle_start}\n")

        self.ingest_buffer.append((s, d, t))

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def batchAddEdges(self, edges):
        """Insert a list of *(src, dst, timestamp)* tuples into the graph."""
        for s, d, t in edges:
            self.timed_list.append(s, d, t)
            self.edge_count[(s, d)] += 1
            self.graph.add_edge(s, d)
            if self.edge_count[(s, d)] == 1:
                self.buckets.addEdge(t)

    # ------------------------------------------------------------------
    # Shedding strategies
    # ------------------------------------------------------------------

    def randomShed(self, count):
        """Remove *count* edges chosen uniformly at random."""
        candidates = []
        curr = self.timed_list.head
        while curr:
            candidates.append(curr)
            curr = curr.next

        count = min(count, len(candidates))
        if count == 0:
            return 0

        to_remove = random.sample(candidates, count)
        for node in to_remove:
            self.timed_list.remove_node(node)
            self.removeEdge(node.src, node.dst)
        return count

    def modifiedSpectralSparsity(self, s):
        """Probabilistically shed edges in the oldest slide.

        Keep probability  ``p = davg * s / min(out_deg(src), in_deg(dst))``.
        Higher *s* keeps more edges.
        """
        davg = getAverageDegree(self.graph, len(self.edge_count))
        curr = self.timed_list.head
        while curr and curr.t < self.window_start + self.slide:
            denom = min(
                self.graph.out_degree(curr.src),
                self.graph.in_degree(curr.dst),
            )
            p = davg * s / denom if denom > 0 else 0
            nxt = curr.next
            if random.random() >= p:
                self.timed_list.remove_node(curr)
                self.removeEdge(curr.src, curr.dst)
            curr = nxt

    # ------------------------------------------------------------------
    # Algorithm execution
    # ------------------------------------------------------------------

    def runAlgo(self, snapshot):
        """Execute the configured algorithm and append results to the log."""
        start = time.perf_counter()
        result = self.algo(snapshot)
        end = time.perf_counter()

        with open(self.algo_log, "a") as f:
            f.write(f"{start},{end},{end - start},{result}\n")
        return result

    # ------------------------------------------------------------------
    # Window maintenance
    # ------------------------------------------------------------------

    def shiftWindow(self, t):
        """Advance the window so that *t* falls inside it.

        Returns the number of expired edges.
        """
        self.window_start = (
            self.base_time
            + math.ceil(
                (t - self.base_time - self.window_size + 1) / self.slide
            )
            * self.slide
        )
        self.window_end = self.window_start + self.window_size

        count = 0
        while self.timed_list.head and self.timed_list.head.t < self.window_start:
            s, d, _ = self.timed_list.popleft()
            self.removeEdge(s, d)
            count += 1
        return count

    def removeEdge(self, s, d):
        """Decrement the multi-edge counter; remove from graph when zero."""
        self.edge_count[(s, d)] -= 1
        if self.edge_count[(s, d)] == 0:
            self.graph.remove_edge(s, d)
            if s in self.graph and self.graph.degree(s) == 0:
                self.graph.remove_node(s)
            if d in self.graph and self.graph.degree(d) == 0:
                self.graph.remove_node(d)
            del self.edge_count[(s, d)]
