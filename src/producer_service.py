import argparse
import time

import redis


MAX_QUEUE_SIZE = 500000
QUEUE_POLL_INTERVAL_SECONDS = 0.05

# Hard-coded producer batch sizing profile.
# Start in REST_BATCH_SIZE, switch to SPIKE_BATCH_SIZE after SPIKE_START_SECONDS,
# then switch back to REST_BATCH_SIZE after SPIKE_END_SECONDS.
REST_BATCH_SIZE = 45
SPIKE_BATCH_SIZE = 190
SPIKE_START_SECONDS = 50
SPIKE_END_SECONDS = 80


def get_target_batch_size(elapsed_seconds: float) -> int:
    if SPIKE_START_SECONDS <= elapsed_seconds < SPIKE_END_SECONDS:
        return SPIKE_BATCH_SIZE
    return REST_BATCH_SIZE


def flush_batch(client: redis.Redis, redis_key: str, batch: list[str]) -> int:
    if not batch:
        return 0
    client.rpush(redis_key, *batch)
    sent_now = len(batch)
    batch.clear()
    return sent_now


def wait_for_queue_capacity(client: redis.Redis, redis_key: str) -> None:
    while client.llen(redis_key) >= MAX_QUEUE_SIZE:
        time.sleep(QUEUE_POLL_INTERVAL_SECONDS)


def run_producer(args):
    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    sent = 0
    next_report = 50000
    start = time.monotonic()
    batch: list[str] = []
    current_batch_size = get_target_batch_size(0.0)

    try:
        print(
            "Producer startup | "
            f"graph_dir={args.graph_dir} | "
            f"redis_url={args.redis_url} | "
            f"redis_key={args.redis_key} | "
            f"max_runtime={args.max_runtime}s | "
            f"rest_batch_size={REST_BATCH_SIZE} | "
            f"spike_batch_size={SPIKE_BATCH_SIZE} | "
            f"spike_start={SPIKE_START_SECONDS}s | "
            f"spike_end={SPIKE_END_SECONDS}s | "
            f"send_end_sentinel={args.send_end_sentinel} | "
            f"clear_key_on_start={args.clear_key_on_start}"
        )

        if args.clear_key_on_start:
            client.delete(args.redis_key)
            print(f"Cleared Redis key '{args.redis_key}' before publishing")

        with open(args.graph_dir, "r") as fh:
            for raw in fh:
                if args.max_runtime > 0 and (time.monotonic() - start) >= args.max_runtime:
                    print(f"Producer timed out after {args.max_runtime} seconds")
                    break

                edge = raw.strip()
                if not edge:
                    continue

                elapsed = time.monotonic() - start
                target_batch_size = get_target_batch_size(elapsed)
                if target_batch_size != current_batch_size:
                    current_batch_size = target_batch_size
                    print(
                        f"Switched producer batch size to {current_batch_size} after {elapsed:.1f}s"
                    )

                batch.append(edge)
                if len(batch) >= current_batch_size:
                    wait_for_queue_capacity(client, args.redis_key)
                    sent += flush_batch(client, args.redis_key, batch)
                    while sent >= next_report:
                        print(f"Published {next_report} edges")
                        next_report += 50000

        wait_for_queue_capacity(client, args.redis_key)
        sent += flush_batch(client, args.redis_key, batch)
        while sent >= next_report:
            print(f"Published {next_report} edges")
            next_report += 50000

        if args.send_end_sentinel:
            wait_for_queue_capacity(client, args.redis_key)
            client.rpush(args.redis_key, args.end_sentinel)
            print("Published end sentinel")

        print(f"Producer finished. Total edges published: {sent}")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Publish edge stream to Redis")
    parser.add_argument("--graph-dir", type=str, required=True, help="Path to graph edge file")
    parser.add_argument("--redis-url", type=str, default="redis://redis:6379/0", help="Redis connection URL")
    parser.add_argument("--redis-key", type=str, default="edges", help="Redis list key for edge messages")
    parser.add_argument(
        "--send-end-sentinel",
        action="store_true",
        help="Push end sentinel when input file is exhausted",
    )
    parser.add_argument("--end-sentinel", type=str, default="__END__", help="Sentinel payload")
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=0,
        help="Maximum producer runtime in seconds (0 means no limit)",
    )
    parser.add_argument(
        "--redis-batch-size",
        type=int,
        default=REST_BATCH_SIZE,
        help="Legacy fixed batch size; kept for compatibility",
    )
    parser.add_argument(
        "--clear-key-on-start",
        action="store_true",
        help="Clear the Redis queue key before producing new edges",
    )
    args = parser.parse_args()

    if args.redis_batch_size <= 0:
        raise ValueError("--redis-batch-size must be greater than 0")
    if REST_BATCH_SIZE <= 0 or SPIKE_BATCH_SIZE <= 0:
        raise ValueError("REST_BATCH_SIZE and SPIKE_BATCH_SIZE must be greater than 0")
    if SPIKE_END_SECONDS <= SPIKE_START_SECONDS:
        raise ValueError("SPIKE_END_SECONDS must be greater than SPIKE_START_SECONDS")

    run_producer(args)


if __name__ == "__main__":
    main()
