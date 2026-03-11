import time

import networkx as nx
from window_manager import WindowManager
import asyncio
from producer_sim import produce, Edge

DATA = "../data/test_graph.txt" # "../data/higgs-activity_time_postprocess.txt"

async def window_trigger(wm: WindowManager, slide: float, base: float):
    """Periodically fires the window function every `slide` real seconds."""
    tick = 1
    while True:
        next_fire = base + tick * slide
        now = time.perf_counter()
        await asyncio.sleep(max(0, next_fire - now))

        close_time = base + tick * slide  # exact boundary, not perf_counter()
        wm.runCompleteWindow(close_time)
        tick += 1

async def main():
    print("hello world")
    # f = open("../data/test_graph.txt", "r")
    g = nx.DiGraph()
    algorithm = nx.pagerank # nx.betweenness_centrality # nx.k_core
    SPEED = 1.0 # seconds between edge arrivals (process time)
    WINDOW_SIZE = 10 # in dataset timestamp units
    SLIDE = 5 # in dataset timestamp units
    base = time.perf_counter()
    wm = WindowManager(WINDOW_SIZE, SLIDE, g, algorithm, base_time=base) # window size = 1000, slide = 500
    wm.warmStart()
    
    trigger_task = asyncio.create_task(window_trigger(wm, SLIDE, base))

    print("start processing edges")
    counter = 1
    async for line in produce(DATA, speed=SPEED):
        print(counter)
        wm.addEdge(line.src, line.dst, time.perf_counter())
        # print(f"Edge added: {line.src} -> {line.dst}, time: {line.ts}, result: {result}")
        counter += 1
        if counter >= 100:
            print("Processed 100 edges, stopping.")
            break
    
    trigger_task.cancel()
    try:
        await trigger_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())