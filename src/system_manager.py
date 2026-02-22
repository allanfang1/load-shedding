
import time
import networkx as nx

# TODO so in theory this class will monitor the system status and apply the correct load shedding accordingly
class SystemManager:
    def __init__(self, graph, algo, window_size, slide):
        self.graph = graph
        self.algo = algo
        self.window_size = window_size
        self.slide = slide  # seconds available to process one window

        self.cpu_rate = 0       # cycles per second (measured via calibration)
        self.cost_per_edge = 0  # cycles per edge (fitted from profiling)
        self.max_edges_per_window = 0  # derived: max edges algo can handle in one slide