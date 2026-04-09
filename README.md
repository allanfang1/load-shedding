# Load Shedding (Containerized)

This project now supports running edge production and window processing in separate containers.

## Architecture

- `producer` container: reads graph edges from file and pushes each line into Redis.
- `main` container: consumes edges from Redis and runs the sliding-window processing pipeline.
- `redis` container: message buffer between producer and main.

`main.py` is Redis-only and does not read graph files directly.

## Run with Docker Compose

From the `load-shedding` folder:

```bash
docker compose up --build
```

To control how long all services stay up, set `STACK_RUNTIME_SECONDS`:

```bash
STACK_RUNTIME_SECONDS=180 docker compose up --build
```

On PowerShell:

```powershell
$env:STACK_RUNTIME_SECONDS = "180"
docker compose up --build --abort-on-container-exit --exit-code-from main
```

Using `--abort-on-container-exit --exit-code-from main` ensures all services stop when `main` exits at the configured runtime.

For higher throughput, tune batching with:

- `PRODUCER_BATCH_SIZE` (default `4000`)
- `CONSUMER_DRAIN_BATCH_SIZE` (default `4000`)

PowerShell example:

```powershell
$env:STACK_RUNTIME_SECONDS = "120"
$env:PRODUCER_BATCH_SIZE = "4000"
$env:CONSUMER_DRAIN_BATCH_SIZE = "4000"
docker compose up --build --abort-on-container-exit --exit-code-from main
```

Output rows are written to `tests/results.csv` (mounted from host).

## Notes

- The producer sends `__END__` when done, so the Redis consumer can stop gracefully.

