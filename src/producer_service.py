import argparse
import time

import redis


def run_producer(args):
    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    sent = 0
    start = time.monotonic()

    try:
        with open(args.graph_dir, "r") as fh:
            for raw in fh:
                if args.max_runtime > 0 and (time.monotonic() - start) >= args.max_runtime:
                    print(f"Producer timed out after {args.max_runtime} seconds")
                    break

                edge = raw.strip()
                if not edge:
                    continue

                client.rpush(args.redis_key, edge)
                sent += 1
                if sent % 50000 == 0:
                    print(f"Published {sent} edges")

        if args.send_end_sentinel:
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
    args = parser.parse_args()

    run_producer(args)


if __name__ == "__main__":
    main()
