
import math
import time
import networkx as nx
from budget import Budget
from welford_variance import WelfordVariance

class TimeCard:
    def __init__(self, start = time.perf_counter(), end = None, count = None):   
        self.start = start
        self.end = end
        self.count = count
    
    def __init__(self, fn, count = None):
        self.start = time.perf_counter()
        result = fn()
        self.end = time.perf_counter()
        self.count = count
    
    def __bool__(self):
        return (
            self.start is not None and
            self.end is not None and
            self.count is not None
        )
    
    def time_function(self, fn):
        self.start = time.perf_counter()
        result = fn()
        self.end = time.perf_counter()
        return result

# TODO so in theory this class will monitor the system status and apply the correct load shedding accordingly
class SystemManager:
    def __init__(self, window_size, slide):
        self.log_path = "system_manager_log.txt"

        self.welfords_cpe_algo = WelfordVariance()  # for tracking algo time variance
        # self.welfords_cpe_shed = WelfordVariance()  # pretend shed costs same as expiry
        self.welfords_cpe_expiry = WelfordVariance()  # for tracking expiry time variance
        self.welfords_cpe_ingest = WelfordVariance()  # for tracking ingest time variance
        # self.welfords_cpe_peek = WelfordVariance()  # TODO let's pretend peek is free for now

        self.window_size = window_size  # seconds
        self.slide = slide  # seconds available to process one window

        self.algo_card = None
        self.shed_card = None
        self.expiry_card = None
        self.ingest_card = None

        self.budget = Budget() # TODO 

        with open(self.log_path, "w") as f:
            f.write(
                "window_count, "
                "algo_time, "
                "algo_mean, "
                "ingest_time, "
                "ingest_mean, "
                "expiry_time, "
                "expiry_mean, "
                "shed_time\n"
            )

    def get_max_edges_per_window(self, alpha, n_slide, n_buffer, n_window): # TODO we have a budget system now, it's a little weird because I need to subtract hmmmmmmmmmm Now I'm confused
        """Calculate max edges per window using Cantelli's inequality.
        Parameters
        ----------
        alpha : float
            Desired upper bound on the probability P(edge processing cost > estimate) = alpha, (0 < alpha < 1).
        """
        self.budget.set_headroom(0.01) # TODO
        self.budget.set_expire(self.cantellis_inequality(self.welfords_cpe_expiry, alpha), n_slide)
        self.budget.set_ingest(self.cantellis_inequality(self.welfords_cpe_ingest, alpha), n_buffer)
        

        total_budget = self.window_size - headroom
        budget  = total_budget - expected_expiry_cost - expected_sparsify_cost - expected_algo_cost
                = total_budget - 
        expected_expiry_cost = self.welfords_cpe_expiry.get_mean() * I know how many edges are leaving
        expected_sparsify_cost = self.welfords_cpe_shed.get_mean() * Do i know how many edges I have right now?
        expected_algo_cost = self.welfords_cpe_algo.get_mean() * but this is what I'm allotting budget for,


        k = math.sqrt(self.welfords_cpe.get_variance() * (1-alpha)/alpha)
        max_cpe = self.welfords_cpe.get_mean() + k
        result = math.floor(self.window_size / max_cpe) if max_cpe > 0 else float('inf')


        if result <= 0:
            raise ValueError("Calculated max edges per window is non-positive.")
        return result # cantelli's inequality
    
    def cantellis_inequality(self, distribution: WelfordVariance, alpha: float):
        """Calculate max edges per window using Cantelli's inequality.
        Parameters
        ----------
        alpha : float
            Desired upper bound on the probability P(edge processing cost > estimate) = alpha, (0 < alpha < 1).
        """
        k = math.sqrt(distribution.get_variance() * (1-alpha)/alpha)
        max_cpe = distribution.get_mean() + k
        result = math.floor(self.window_size / max_cpe) if max_cpe > 0 else float('inf')
        return result

    def get_max_edges_per_window_naive(self):
        result = math.floor(self.window_size / self.welfords_cpe.get_mean()) if self.welfords_cpe.count > 0 else float('inf')
        if result <= 0:
            raise ValueError("Calculated max edges per window is non-positive.")
        return result

    def update_cycle_cost(self):
        if not self.algo_card or not self.shed_card or not self.expiry_card or not self.ingest_card:
            raise ValueError("Time card(s) not initialized properly.")
        if self.algo_card.count > 0: 
            self.welfords_cpe_algo.add_var((self.algo_card.end - self.algo_card.start) / 
                                        self.algo_card.count)
        if self.expiry_card.count > 0:
            self.welfords_cpe_expiry.add_var((self.expiry_card.end - self.expiry_card.start) / 
                                        self.expiry_card.count)
        if self.ingest_card.count > 0:
            self.welfords_cpe_ingest.add_var((self.ingest_card.end - self.ingest_card.start) /
                                        self.ingest_card.count)            
        
        with open(self.log_path, "a") as f:
            f.write(
                f"{self.welfords_cpe_algo.count}, " # number of windows processed
                f"{self.algo_card.end - self.algo_card.start}, " # total time to process last window
                f"{self.welfords_cpe_algo.get_mean()}, "
                f"{self.ingest_card.end - self.ingest_card.start}, " # time spent on ingest
                f"{self.welfords_cpe_ingest.get_mean()}, "
                f"{self.expiry_card.end - self.expiry_card.start}, " # time spent on expiry
                f"{self.welfords_cpe_expiry.get_mean()}, "
                f"{self.shed_card.end - self.shed_card.start}\n " # time spent on sparsify
            )

    # someone needs to provide this with how long it took to process the last window, the headroom, how to calculate the sparsification parameter
    # for random, it is formulated as: 
