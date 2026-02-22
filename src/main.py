import networkx as nx
from window_manager import WindowManager
import asyncio
from producer_sim import produce, Edge

DATA = "../data/test_graph.txt" # "../data/higgs-activity_time_postprocess.txt"

async def main():
    print("hello world")
    # f = open("../data/test_graph.txt", "r")
    g = nx.DiGraph()
    algorithm = nx.pagerank # nx.betweenness_centrality # nx.k_core
    wm = WindowManager(10, 5, g, algorithm) # window size = 1000, slide = 500

    print("start processing edges")
    counter = 0
    async for line in produce(DATA, speed=1.0):
        print(counter)
        result = wm.addEdge(line.src, line.dst, line.ts)
        print(f"Edge added: {line.src} -> {line.dst}, time: {line.ts}, result: {result}")
        counter += 1
        if counter >= 100:
            print("Processed 100 edges, stopping.")
            break

if __name__ == "__main__":
    asyncio.run(main())