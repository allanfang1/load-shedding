import argparse
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

async def producer_spike(queue: asyncio.Queue, args):
    """Simulates edge arrivals and puts them in the queue."""
    async for edge in produce(
        args.graph_dir, 
        idle_rate=args.idle_rate, 
        spike_rate=args.spike_rate, 
        spike_start=time.perf_counter() + args.spike_time, 
        spike_duration=args.spike_duration):
        await queue.put(edge)

async def producer_sustain(queue: asyncio.Queue, args):
    """Simulates edge arrivals and puts them in the queue."""
    async for edge in produce(args.graph_dir, idle_rate=args.idle_rate):
        await queue.put(edge)

async def consumer(queue: asyncio.Queue, wm: WindowManager):
    """Consumes edges from the queue and adds them to the WindowManager."""
    while True:
        edge = await queue.get()
        if edge is None:  # Sentinel value to stop the consumer
            break
        wm.addEdge(edge)

def common_data_args():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--graph-dir", type=str, default=None, help="Path to graph edge file", required=True)
    parent.add_argument("--output-dir", type=str, default="output.csv", help="Directory to save collected data or trained model")
    parent.add_argument("--k", type=int, default=15, help="Value of k for top-k metrics")
    parent.add_argument("--total-runtime", type=int, default=5000, help="Total runtime for producer in seconds")
    parent.add_argument("--idle-rate", type=float, default=0.001, help="Sleep seconds between edges when simulating idle periods")
    parent.add_argument("--window-size", type=float, default=10.0, help="Size of the sliding window in seconds")
    parent.add_argument("--slide", type=float, default=5.0, help="Slide interval for the window in seconds")
    parent.add_argument("--model-dir", type=str, default=None, help="Directory containing trained model")
    return parent

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph algorithm runtime predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_spike = sub.add_parser("spike", parents=[common_data_args()], help="Run a singular spike test")
    p_spike.add_argument("--spike_time", type=float, default=10.0, help="Time of the spike from start time in seconds")
    p_spike.add_argument("--spike_duration", type=float, default=5.0, help="Duration of the spike in seconds")
    p_spike.add_argument("--spike_rate", type=float, default=0.001, help="Edge arrival rate during the spike in sleep seconds between edges")

    sub.add_parser("sustain", parents=[common_data_args()], help="Run sustained-rate test")
    return parser

async def main():
    parser = build_parser()
    args = parser.parse_args()
    np.random.seed(42)

    if not os.path.isfile(args.graph_dir):
        print(f"Graph file {args.graph_dir} does not exist.")
        return

    cmd_map = {
        "spike": producer_spike,
        "sustain": producer_sustain,
    }

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

    wm = WindowManager(args.window_size, args.slide, g, algorithm, base_time=base, predictor=predictor)
    wm.warmStart()

    window_lock = asyncio.Lock()
    trigger_task = asyncio.create_task(window_trigger(wm, args.slide, base, window_lock))

    queue = asyncio.Queue()
    print("start processing edges")
    producer_task = asyncio.create_task(cmd_map[args.command](queue, args))
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