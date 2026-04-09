import argparse
import csv
import time
import os

import networkx as nx
import numpy as np
from window_manager import WindowManager
import asyncio
from redis import asyncio as redis

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

async def redis_consumer(
    wm: WindowManager,
    redis_url: str,
    redis_key: str,
    end_sentinel: str,
    block_timeout: int = 1,
):
    """Consumes edges from a Redis list and adds them to the WindowManager."""
    client = redis.from_url(redis_url, decode_responses=True)
    try:
        while True:
            message = await client.blpop(redis_key, timeout=block_timeout)
            if message is None:
                await asyncio.sleep(0)
                continue

            _, edge = message
            if edge == end_sentinel:
                print("Received end sentinel from Redis producer")
                break
            wm.addEdge(edge)
    finally:
        await client.aclose()

async def pipeline(args):
    """Main pipeline that consumes edges from Redis and runs the window manager."""
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

    print("start processing edges from Redis")
    consumer_task = asyncio.create_task(
        redis_consumer(
            wm=wm,
            redis_url=args.redis_url,
            redis_key=args.redis_key,
            end_sentinel=args.end_sentinel,
            block_timeout=args.redis_block_timeout,
        )
    )
    
    try:
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
    parser.add_argument("--output-dir", type=str, default="output.csv", help="Directory to save collected data or trained model")
    parser.add_argument("--k", type=int, default=15, help="Value of k for top-k metrics")
    parser.add_argument("--total-runtime", type=int, default=5000, help="Total runtime for producer in seconds")
    parser.add_argument("--window-size", type=float, default=10.0, help="Size of the sliding window in seconds")
    parser.add_argument("--slide", type=float, default=5.0, help="Slide interval for the window in seconds")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing trained model")
    parser.add_argument("--redis-url", type=str, default="redis://redis:6379/0", help="Redis connection URL")
    parser.add_argument("--redis-key", type=str, default="edges", help="Redis list key for edge messages")
    parser.add_argument("--end-sentinel", type=str, default="__END__", help="Sentinel payload used to stop Redis consumer")
    parser.add_argument(
        "--redis-block-timeout",
        type=int,
        default=1,
        help="BLPOP timeout in seconds while waiting for Redis messages",
    )
    args = parser.parse_args()
    np.random.seed(42)

    try:
        await asyncio.wait_for(pipeline(args), timeout=args.total_runtime)
    except asyncio.TimeoutError:
        print(f"Stopped after {args.total_runtime} seconds.")

if __name__ == "__main__":
    asyncio.run(main())