import time
import os

import networkx as nx
from window_manager import WindowManager
import asyncio
from producer_sim import produce

from modelling_s.runtime_predictor import RuntimePredictor

DATA = "../data/ldbc-sf10-updatestream_postprocess.txt" # "../data/higgs-activity_time_postprocess.txt" D:\Desktop\reading\M2\loadshedding\load-shedding\data\ldbc-sf10-updatestream_postprocess.txt
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models") # Directory where trained models are saved/loaded

async def window_trigger(
    wm: WindowManager,
    slide: float,
    base: float,
    window_lock: asyncio.Lock,
):
    """Periodically fires the window function every `slide` real seconds."""
    tick = 1
    while True:
        next_fire = base + tick * slide
        now = time.perf_counter() # might be better to use loop.time()
        await asyncio.sleep(max(0, next_fire - now))

        close_time = base + tick * slide  # exact boundary, not perf_counter()
        async with window_lock:
            await asyncio.to_thread(wm.runCompleteWindow, close_time)
        tick += 1

async def producer(queue: asyncio.Queue):
    """Simulates edge arrivals and puts them in the queue."""
    async for edge in produce(DATA):
        await queue.put(edge)

async def consumer(queue: asyncio.Queue, wm: WindowManager):
    """Consumes edges from the queue and adds them to the WindowManager."""
    while True:
        edge = await queue.get()
        if edge is None:  # Sentinel value to stop the consumer
            break
        wm.addEdge(edge)

async def main():
    g = nx.DiGraph() # we are using a digraph
    algorithm = nx.pagerank # nx.betweenness_centrality # nx.k_core
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

    window_lock = asyncio.Lock()
    
    trigger_task = asyncio.create_task(window_trigger(wm, SLIDE, base, window_lock))

    queue = asyncio.Queue()
    print("start processing edges")
    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(consumer(queue, wm))
    
    try:
        await producer_task  # wait until done
        await queue.put(None)  # signal consumer to stop
        await consumer_task
    finally:
        trigger_task.cancel()
        try:
            await trigger_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())