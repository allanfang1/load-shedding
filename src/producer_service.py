import argparse
import time

import redis


MAX_QUEUE_SIZE = 500000
QUEUE_POLL_INTERVAL_SECONDS = 0.05


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

    try:
        print(
            "Producer startup | "
            f"graph_dir={args.graph_dir} | "
            f"redis_url={args.redis_url} | "
            f"redis_key={args.redis_key} | "
            f"max_runtime={args.max_runtime}s | "
            f"batch_size={args.redis_batch_size} | "
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

                batch.append(edge)
                if len(batch) >= args.redis_batch_size:
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
        default=2000,
        help="Number of edges to publish per Redis RPUSH call",
    )
    parser.add_argument(
        "--clear-key-on-start",
        action="store_true",
        help="Clear the Redis queue key before producing new edges",
    )
    args = parser.parse_args()

    if args.redis_batch_size <= 0:
        raise ValueError("--redis-batch-size must be greater than 0")

    run_producer(args)


if __name__ == "__main__":
    main()
