from collections import deque, defaultdict
import math
import time
import networkx as nx
import asyncio

class WindowManager:
    def __init__(self, window_size, slide, graph, algo, start_time=0):
        if slide > window_size:
            raise ValueError("slide should be less than or equal to window_size")
        self.window_size = window_size
        self.slide = slide
        self.start_time = start_time
        self.window_start = start_time
        self.window_end = start_time + window_size
        self.adjacency_list = deque()
        self.graph = graph
        self.edge_count = defaultdict(int)
        self.algo = algo
        self.window_log = "timing_log.txt"
        self.algo_log = "algo_log.txt"
        self.window_log_start_time = time.perf_counter()

    def addEdge(self, s, d, t):
        result = None
        if t < self.window_start:
            return
        if t > self.window_end:
            temp = time.perf_counter()
            with open(self.window_log, "a") as f:
                f.write(
                    f"{self.window_log_start_time}, "
                    f"{temp}, "
                    f"{temp - self.window_log_start_time}\n"
                )
            self.window_log_start_time = temp
            snapshot = self.sparsify() # TODO sparsifier
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self.runAlgo, snapshot)
            self.shiftWindow(t)
        self.adjacency_list.append((s, d, t))
        self.edge_count[(s, d)] += 1
        self.graph.add_edge(s, d)
        print(f"Edge added: {s} -> {d}, time: {t}")
        print(f"Edge count: {self.edge_count}")
        return result
    
    def sparsify(self): # TODO implement different sparsification strategies
        self.graph
        self.adjacency_list
        return self.graph.copy()

    def runAlgo(self, snapshot):
        start_time = time.perf_counter()
        result = self.algo(snapshot)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        with open(self.algo_log, "a") as f:
            f.write(
                f"{start_time}, "
                f"{end_time}, "
                f"{elapsed}\n"
            )
        # TODO: store or print results
    
    def shiftWindow(self, t):
        self.window_start = self.start_time + math.ceil((t - self.start_time - self.window_size) / self.slide) * self.slide
        self.window_end = self.window_start + self.window_size
        while self.adjacency_list and self.adjacency_list[0][2] < self.window_start:
            s, d, edge_t = self.adjacency_list.popleft()
            self.edge_count[(s, d)] -= 1
            if self.edge_count[(s, d)] == 0:
                print(f"Edge removed from graph: {s} -> {d}, time: {edge_t}")
                self.graph.remove_edge(s, d)
                if self.graph.degree(s) == 0:
                    self.graph.remove_node(s)
                    print(f"Node removed: {s}")
                if self.graph.degree(d) == 0:
                    self.graph.remove_node(d)
                    print(f"Node removed: {d}")
                del self.edge_count[(s, d)]
            print(f"Edge removed: {s} -> {d}, time: {edge_t}")
        print(f"Window moved to [{self.window_start}, {self.window_end}]")
    
    def getEdgeCount(self):
        return sum(self.edge_count.values())