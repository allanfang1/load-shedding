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

## Run `modelling_s` in the same container image

Use the `modelling` service when you want training data collection or `run-all` under the exact same Python/runtime image as `main`.

Build once:

```bash
docker compose build main modelling
```

Collect training data (example):

```bash
docker compose run --rm modelling python -m modelling_s.main collect --sample-file /data/higgs-activity_time_postprocess.txt --algo pagerank --out /app/tests/timings.csv --num-snapshots 15 --max-edges 5000
```

Run end-to-end collect + train (example):

```bash
docker compose run --rm modelling python -m modelling_s.main run-all --sample-file /data/higgs-activity_time_postprocess.txt --algo pagerank --model-dir /app/src/models --out /app/tests/timings.csv
```

Because of mounted volumes:

- input data is read from `./data` via `/data`
- collected CSVs are persisted to `./tests`
- trained artifacts are persisted to `./src/models`

If you prefer, you can also run the same command through the `main` service image:

```bash
docker compose run --rm main python -m modelling_s.main run-all --sample-file /data/higgs-activity_time_postprocess.txt --algo pagerank --model-dir /app/src/models --out /app/tests/timings.csv
```

## Notes

- The producer sends `__END__` when done, so the Redis consumer can stop gracefully.

