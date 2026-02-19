import math
import networkit as nk

class TimedEdge:
    def __init__(self, s, d, t):
        self.s = s
        self.d = d
        self.t = t
        self.next = None

class WindowManager:
    def __init__(self, window_size, slide, graph, algo, start_time=0):
        if slide > window_size:
            raise ValueError("slide should be less than or equal to window_size")
        self.window_size = window_size
        self.slide = slide
        self.start_time = start_time
        self.window_start = start_time
        self.window_end = start_time + window_size
        self.edges = GraphLinkedList()
        self.graph = graph
        self.algo = algo
    
    def addEdge(self, s, d, t):
        if t < self.window_start:
            return
        if t > self.window_end:
            self.shiftWindow(t)
        print(f"Adding edge: {s} -> {d}, time: {t}")
        self.edges.addEdge(s, d, t)
        self.graph.addEdge(s, d, addMissing=True)
        self.algo.update(nk.dynamics.GraphEvent(nk.dynamics.GraphEventType.EDGE_ADDITION, s, d, 1.0))
        print(f"Edge added: {s} -> {d}, time: {t}")
        return self.algo.run()
    
    def shiftWindow(self, t):
        self.window_start = self.start_time + math.ceil((t - self.start_time - self.window_size) / self.slide) * self.slide
        self.window_end = self.window_start + self.window_size
        edgesToDelete = self.edges.deleteEdgesBefore(self.window_start)
        batch = []
        while edgesToDelete is not None:
            self.graph.removeEdge(edgesToDelete.s, edgesToDelete.d)
            batch.append(nk.dynamics.GraphEvent(nk.dynamics.GraphEventType.EDGE_DELETION, edgesToDelete.s, edgesToDelete.d, 1.0))
            edgesToDelete = edgesToDelete.next
        self.algo.updateBatch(batch)
        print(f"Window moved to [{self.window_start}, {self.window_end}]")

class GraphLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def addEdge(self, s, d, t):
        edge = TimedEdge(s, d, t)
        if self.head is None:
            self.head = edge
            self.tail = edge
        else:
            self.tail.next = edge
            self.tail = edge
    
    def deleteEdgesBefore(self, time):
        if self.head is None or self.head.t >= time:
            return None
        temp = self.head
        curr = self.head
        while self.head is not None and self.head.t < time:
            curr = self.head
            self.head = self.head.next
        if self.head is None:
            self.tail = None
        curr.next = None
        return temp

def main():
    print("hello world")
    f = open("../data/higgs-activity_time_postprocess.txt", "r")
    g = nk.Graph()
    dynAPSP = nk.distance.DynAPSP(g)
    dynAPSP.run()
    wm = WindowManager(1000, 500, g, dynAPSP) # window size = 1000, slide = 500

    print("start processing edges")
    counter = 0
    for line in f:
        print(counter);
        s, d, _, t = line.strip().split()
        result = wm.addEdge(int(s), int(d), int(t))
        print(f"Edge added: {s} -> {d}, time: {t}, result: {result}")
        counter += 1
        if counter >= 1000:
            print("Processed 1000 edges, stopping.")
            break

if __name__ == "__main__":
    main()