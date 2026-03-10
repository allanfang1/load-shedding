from collections import deque, defaultdict
import math
import random
import time
import networkx as nx
import asyncio
from sparsifiers import davgSparsify
from buckets import Buckets
from timed_linkedlist import TimedLL, TimedLLNode
from helper import getAverageDegree
from system_manager import SystemManager, TimeCard

class WindowManager:
    def __init__(self, window_size, slide, graph, algo, base_time=0):
        """
        Parameters
        ----------
        window_size : int
            Size of the sliding window in **dataset timestamp units**.
        slide : int
            Slide interval in **dataset timestamp units**.
        base_time : int
            First expected timestamp in dataset units.
        """
        if slide > window_size:
            raise ValueError("slide should be less than or equal to window_size")
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

        self.sm = SystemManager(window_size, slide)

        random.seed(42)
        self.start_time_system = time.perf_counter()
        self.warmStart()
    
    def warmStart(self):
        self.runAlgo(self.graph)

    def addEdge(self, s, d, t):
        """Add edge (s, d) with timestamp t to the current window, shifting the window if necessary.
        Includes timing for:
            - Incremental edge addition is NOT timed 
            - Burst edge expiration
            - Algorithm execution
            - Sparsification
        """
        if t < self.window_start:
            return
        if t >= self.window_end:
            temp = time.perf_counter()
            with open(self.window_log, "a") as f:
                f.write(
                    f"{self.start_time_system}, "
                    f"{temp}, "
                    f"{temp - self.start_time_system}\n"
                )
            self.start_time_system = temp

            ### START INGEST TIME ###
            self.sm.ingest_card = TimeCard(lambda: self.batchAddEdges(self.ingest_buffer), len(self.ingest_buffer))
            self.ingest_buffer = []
            ### END INGEST TIME ###

            ### START SPARSIFY TIME ###
            self.sm.shed_card = TimeCard(lambda: None, 1) # reset time card for next window # TODO count should be number of edges pruned or processed,
            ### END SPARSIFY TIME ###

            ### START ALGO TIME ###
            self.sm.algo_card = TimeCard(lambda: self.runAlgo(self.graph), self.graph.number_of_edges()) # reset time card for next window
            ### END ALGO TIME ###

            ### START EXPIRY TIME ###
            self.sm.expiry_card = TimeCard() # reset time card for next window
            self.sm.expiry_card.count = self.shiftWindow(t)
            self.buckets.removeBefore(t)
            ### END EXPIRY TIME ###
            self.sm.expiry_card.end = time.perf_counter()
            
            self.sm.update_cycle_cost()
        
        self.ingest_buffer.append((s, d, t))

        print(f"Bucket: {self.buckets.getCount(t)} edges in current bucket")
        print(f"Edge added: {s} -> {d}, time: {t}")
        print(f"Edge count: {self.edge_count}")
        print(f"Timed list size: {self.timed_list.size}")
    
    def batchAddEdges(self, edges):
        for s, d, t in edges:
            self.timed_list.append(s, d, t)
            self.edge_count[(s, d)] += 1
            self.graph.add_edge(s, d)
            if self.edge_count[(s, d)] == 1:
                self.buckets.addEdge(t)

    def sparsify(self): # TODO implement different sparsification strategies
        self.graph
        self.timed_list
        # return davgSparsify(self.graph, self.graph, 1, self.getAverageDegree())
        return self.graph.copy()
    
    def modifiedSpectralSparsity(self, s):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
        davg = self.getAverageDegree(self.graph, len(self.edge_count))
        curr = self.timed_list.head
        while curr and curr.t < self.window_start + self.slide:
            denom = min(self.graph.out_degree(curr.src), self.graph.in_degree(curr.dst))
            p = davg * s / denom if denom > 0 else 0

            temp = curr.next
            if random.random() >= p:
                self.timed_list.remove_node(curr) # remove edge from timed_list
                self.removeEdge(curr.src, curr.dst)
            curr = temp
    
    def randomSparsity(self, s): # TODO
        pass

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
        return result
        # TODO: store or print results
    
    def shiftWindow(self, t):
        self.window_start = self.base_time + math.ceil((t - self.base_time - self.window_size + 1) / self.slide) * self.slide # earliest window that contains the edge, only works for integer timestamps
        self.window_end = self.window_start + self.window_size
        count = 0
        while self.timed_list and self.timed_list.head.t < self.window_start:
            s, d, edge_t = self.timed_list.popleft()
            self.removeEdge(s, d)
            count += 1
            print(f"Edge removed: {s} -> {d}, time: {edge_t}")
        print(f"Window moved to [{self.window_start}, {self.window_end}]")
        return count
    
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