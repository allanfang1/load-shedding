import argparse
import csv
import time
import os
import traceback

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
            try:
                row = await asyncio.to_thread(wm.runCompleteWindow, close_time)
                append_row(row, args.output_dir)
            except Exception as e:
                print(f"Window execution failed at close_time={close_time}: {e}")
                traceback.print_exc()
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
    first_edge_event: asyncio.Event | None = None,
    block_timeout: int = 1,
    drain_batch_size: int = 2000,
):
    """Consumes edges from a Redis list and adds them to the WindowManager."""
    client = redis.from_url(redis_url, decode_responses=True)
    try:
        while True:
            message = await client.blpop(redis_key, timeout=block_timeout)
            if message is None:
                await asyncio.sleep(0)
                continue

            _, first_edge = message
            edges = [first_edge]

            if drain_batch_size > 1:
                drained = await client.lpop(redis_key, drain_batch_size - 1)
                if drained is None:
                    drained_edges = []
                elif isinstance(drained, str):
                    drained_edges = [drained]
                else:
                    drained_edges = drained
                edges.extend(drained_edges)

            for edge in edges:
                if edge == end_sentinel:
                    print("Received end sentinel from Redis producer")
                    return
                wm.addEdge(edge)
                if first_edge_event is not None and not first_edge_event.is_set():
                    first_edge_event.set()
                    print("First edge consumed from Redis; enabling window trigger")
    finally:
        await client.aclose()

async def pipeline(args):
    """Main pipeline that consumes edges from Redis and runs the window manager."""
    g = nx.DiGraph() # we are using a digraph
    algorithm = nx.strongly_connected_components # nx.triangles # nx.pagerank # nx.betweenness_centrality # nx.k_core

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

    wm = WindowManager(
        args.window_size,
        args.slide,
        g,
        algorithm,
        args.k,
        base_time=0.0,
        predictor=predictor,
        headroom_percent=args.headroom,
    )
    wm.warmStart()

    window_lock = asyncio.Lock()
    first_edge_event = asyncio.Event()
    trigger_task = None

    print("start processing edges from Redis")
    consumer_task = asyncio.create_task(
        redis_consumer(
            wm=wm,
            redis_url=args.redis_url,
            redis_key=args.redis_key,
            end_sentinel=args.end_sentinel,
            first_edge_event=first_edge_event,
            block_timeout=args.redis_block_timeout,
            drain_batch_size=args.redis_drain_batch_size,
        )
    )

    first_edge_wait_task = asyncio.create_task(first_edge_event.wait())
    
    try:
        done, _ = await asyncio.wait(
            {consumer_task, first_edge_wait_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if first_edge_wait_task in done and first_edge_event.is_set():
            trigger_base = time.perf_counter()
            wm.base_time = trigger_base
            wm.window_start = trigger_base
            trigger_task = asyncio.create_task(window_trigger(wm, args.slide, trigger_base, window_lock, args))

        await consumer_task
    finally:
        first_edge_wait_task.cancel()
        if trigger_task is not None:
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
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.0,
        help="Headroom as a percentage of slide to subtract from remaining-time budget (0-100)",
    )
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
    parser.add_argument(
        "--redis-drain-batch-size",
        type=int,
        default=2000,
        help="How many additional edges to drain per Redis read cycle",
    )
    args = parser.parse_args()
    np.random.seed(42)

    print(
        "Main startup | "
        f"total_runtime={args.total_runtime}s | "
        f"redis_url={args.redis_url} | "
        f"redis_key={args.redis_key} | "
        f"redis_block_timeout={args.redis_block_timeout}s | "
        f"redis_drain_batch_size={args.redis_drain_batch_size} | "
        f"window_size={args.window_size} | "
        f"slide={args.slide} | "
        f"headroom={args.headroom}% | "
        f"model_dir={args.model_dir}"
    )

    if args.redis_drain_batch_size <= 0:
        raise ValueError("--redis-drain-batch-size must be greater than 0")
    if args.headroom < 0 or args.headroom > 100:
        raise ValueError("--headroom must be in the range [0, 100]")

    try:
        await asyncio.wait_for(pipeline(args), timeout=args.total_runtime)
    except asyncio.TimeoutError:
        print(f"Stopped after {args.total_runtime} seconds.")

if __name__ == "__main__":
    asyncio.run(main())