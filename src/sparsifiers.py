import random

def modifiedSpectralSparsity(self, s):  # for a -> b, davg * s / min(degAout, degBin) where s is provided by the system manager
    davg = self.getAverageDegree()
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