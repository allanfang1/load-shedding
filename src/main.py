import argparse
import csv
import time
import os

import networkx as nx
import numpy as np
from window_manager import WindowManager
import asyncio
from producer_sim import produce

from modelling_s.runtime_predictor import RuntimePredictor

# DATA = "../data/ldbc-sf10-updatestream_postprocess.txt" # "../data/higgs-activity_time_postprocess.txt" D:\Desktop\reading\M2\loadshedding\load-shedding\data\ldbc-sf10-updatestream_postprocess.txt
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "models") # Directory where trained models are saved/loaded

async def window_trigger(
    wm: WindowManager,
    slide: float,
    base: float,
    window_lock: asyncio.Lock,
    args
):
    """Periodically fires the window function every `slide` real seconds."""
    tick = 1
    while True:
        next_fire = base + tick * slide
        now = time.perf_counter() # might be better to use loop.time()
        await asyncio.sleep(max(0, next_fire - now))

        close_time = base + tick * slide  # exact boundary, not perf_counter()
        async with window_lock:
            append_row(await asyncio.to_thread(wm.runCompleteWindow, close_time), args.output_dir)
        tick += 1

def append_row(row: dict, output_dir: str) -> None:
    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    
    file_exists = os.path.exists(output_dir)

    with open(output_dir, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
        print(f"Saved row to {output_dir}")

async def producer(queue: asyncio.Queue, args):
    """Simulates edge arrivals and puts them in the queue."""
    async for edge in produce(args.graph_dir):
        await queue.put(edge)

async def consumer(queue: asyncio.Queue, wm: WindowManager):
    """Consumes edges from the queue and adds them to the WindowManager."""
    while True:
        edge = await queue.get()
        if edge is None:  # Sentinel value to stop the consumer
            break
        wm.addEdge(edge)

async def pipeline(args):
    """Main pipeline to run the producer and consumer with the window manager."""
    g = nx.DiGraph() # we are using a digraph
    algorithm = nx.pagerank # nx.betweenness_centrality # nx.k_core
    base = time.perf_counter()

    # Load trained predictor for load shedding (if available)
    predictor = None
    if args.model_dir:
        if os.path.isdir(args.model_dir):
            try:
                predictor = RuntimePredictor.load(args.model_dir)
                print(f"Loaded runtime predictor for '{predictor.algorithm_name}' from {args.model_dir}")
            except Exception as e:
                print(f"Could not load predictor from {args.model_dir}: {e}")
        else:
            print(f"No trained model found at {args.model_dir} — running without load shedding")

    wm = WindowManager(args.window_size, args.slide, g, algorithm, args.k, base_time=base, predictor=predictor)
    wm.warmStart()

    window_lock = asyncio.Lock()
    trigger_task = asyncio.create_task(window_trigger(wm, args.slide, base, window_lock, args))

    queue = asyncio.Queue()
    print("start processing edges")
    producer_task = asyncio.create_task(producer(queue, args))
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

async def main():
    parser = argparse.ArgumentParser(
        description="Graph algorithm runtime predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph-dir", type=str, default=None, help="Path to graph edge file", required=True)
    parser.add_argument("--output-dir", type=str, default="output.csv", help="Directory to save collected data or trained model")
    parser.add_argument("--k", type=int, default=15, help="Value of k for top-k metrics")
    parser.add_argument("--total-runtime", type=int, default=5000, help="Total runtime for producer in seconds")
    parser.add_argument("--window-size", type=float, default=10.0, help="Size of the sliding window in seconds")
    parser.add_argument("--slide", type=float, default=5.0, help="Slide interval for the window in seconds")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing trained model")
    args = parser.parse_args()
    np.random.seed(42)

    if not os.path.isfile(args.graph_dir):
        print(f"Graph file {args.graph_dir} does not exist.")
        return

    try:
        await asyncio.wait_for(pipeline(args), timeout=args.total_runtime)
    except asyncio.TimeoutError:
        print(f"Stopped after {args.total_runtime} seconds.")

if __name__ == "__main__":
    asyncio.run(main())