import time
import os

import networkx as nx
from window_manager import WindowManager
import asyncio
from producer_sim import produce, Edge

from modelling_s.runtime_predictor import RuntimePredictor

DATA = "../data/test_graph.txt" # "../data/higgs-activity_time_postprocess.txt"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models") # Directory where trained models are saved/loaded

async def window_trigger(wm: WindowManager, slide: float, base: float):
    """Periodically fires the window function every `slide` real seconds."""
    tick = 1
    while True:
        next_fire = base + tick * slide
        now = time.perf_counter() # might be better to use loop.time()
        await asyncio.sleep(max(0, next_fire - now))

        close_time = base + tick * slide  # exact boundary, not perf_counter()
        wm.runCompleteWindow(close_time)
        tick += 1

async def main():
    print("hello world")
    # f = open("../data/test_graph.txt", "r")
    g = nx.DiGraph()
    algorithm = nx.betweenness_centrality # nx.pagerank # nx.k_core
    SPEED = 1.0 # seconds between edge arrivals (process time)
    WINDOW_SIZE = 10 # in system seconds
    SLIDE = 5 # in system seconds
    base = time.perf_counter()

    # Load trained predictor for load shedding (if available)
    predictor = None
    if os.path.isdir(MODEL_DIR):
        try:
            predictor = RuntimePredictor.load(MODEL_DIR)
            print(f"Loaded runtime predictor for '{predictor.algorithm_name}' from {MODEL_DIR}")
        except Exception as e:
            print(f"Could not load predictor from {MODEL_DIR}: {e}")
    else:
        print(f"No trained model found at {MODEL_DIR} — running without load shedding")

    wm = WindowManager(WINDOW_SIZE, SLIDE, g, algorithm, base_time=base, predictor=predictor)
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