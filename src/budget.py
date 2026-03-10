
import math


class Budget:
    def __init__(self, budget: int, ingest: int = 0, expire: int = 0, algo: int = 0, shed: int = 0, headroom: int = 0):
        self.total_budget = budget
        self.ingest = ingest
        self.expire = expire
        self.headroom = headroom

        # self.algo = algo
        # self.shed = shed

    def set_ingest(self, cost_per_edge: int, edge_count: int):
        self.ingest = cost_per_edge * edge_count
    
    def set_expire(self, cost_per_cost: int, edge_count: int):
        self.expire = cost_per_cost * edge_count

    def set_headroom(self, headroom: int):
        self.headroom = headroom
    
    def get_shed_keep_count(self, cost_per_edge_algo: int, edge_count_window: int, edge_count_slide: int, cost_per_edge_expire: int, cost_per_edge_peek: int = 0):
        """
        SOLVE FOR THE POINT WHERE edges to shed (nshed) + edges to keep for algo (nalgo) is the budget constraint:
        Bremaining = B − Bingest − Bexpire − H
        Bremaining = Bshed + Balgo
        Bremaining = (cedgeexpire ∗ nshed + cedgepeek ∗ nslide) + cedgealgo ∗ nalgo
        nshed + nalgo = nwindow
        nalgo = nwindow − nshed
        Bremaining = (cedgeexpire ∗ nshed + cedgepeek ∗ nslide) + cedgealgo ∗ (nwindow − nshed)
        Bremaining = (cedgeexpire − cedgealgo) ∗ nshed + cedgealgo ∗ nwindow + cedgepeek ∗ nslide
        nshed = (Bremaining − cedgealgo ∗ nwindow − cedgepeek ∗ nslide)/(cedgeexpire − cedgealgo)

        if we pretend peek is free, this simplifies to:
        nshed = (Bremaining − cedgealgo ∗ nwindow)/(cedgeexpire − cedgealgo)
        
        Parameters
        ----------
        cost_per_edge_algo : int
            Cost per edge for the algorithm.
        edge_count_window : int
            Number of edges in the current window.
        cost_per_edge_peek : int
            Cost per edge for peeking.
        edge_count_slide : int
            Number of edges in the slide.
        cost_per_edge_expire : int
            Cost per edge for expiration.
        
        Returns
        -------
        shed_count : int
            Number of edges to shed.
        algo_count : int
            Number of edges to keep for the algorithm.
        """
        if cost_per_edge_expire >= cost_per_edge_algo: # TODO put some checks for this
            return 0, edge_count_window
        remaining = self.total_budget - self.ingest - self.expire - self.headroom
        shed_count = (remaining - cost_per_edge_algo * edge_count_window - cost_per_edge_peek * edge_count_slide) / (cost_per_edge_expire - cost_per_edge_algo)
        if shed_count < 0:
            shed_count = 0
        shed_count = math.ceil(shed_count)
        algo_count = edge_count_window - shed_count
        return shed_count, algo_count
        