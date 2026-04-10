#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./run_batch_sweep.sh
#   BATCH_SIZES="500 1000 2000" STACK_RUNTIME_SECONDS=180 ./run_batch_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$ROOT_DIR/tests"
BASE_RESULTS_FILE="$RESULTS_DIR/results.csv"

# Space-separated list of producer batch sizes to test.
BATCH_SIZES="${BATCH_SIZES:-500 1000 2000 4000}"

# Reuse current compose default unless caller overrides.
STACK_RUNTIME_SECONDS="${STACK_RUNTIME_SECONDS:-120}"
CONSUMER_DRAIN_BATCH_SIZE="${CONSUMER_DRAIN_BATCH_SIZE:-4000}"

mkdir -p "$RESULTS_DIR"

for BATCH_SIZE in $BATCH_SIZES; do
  echo "============================================================"
  echo "Running pipeline with PRODUCER_BATCH_SIZE=$BATCH_SIZE"
  echo "STACK_RUNTIME_SECONDS=$STACK_RUNTIME_SECONDS"
  echo "CONSUMER_DRAIN_BATCH_SIZE=$CONSUMER_DRAIN_BATCH_SIZE"
  echo "============================================================"

  rm -f "$BASE_RESULTS_FILE"

  (
    cd "$ROOT_DIR"
    PRODUCER_BATCH_SIZE="$BATCH_SIZE" \
    CONSUMER_DRAIN_BATCH_SIZE="$CONSUMER_DRAIN_BATCH_SIZE" \
    STACK_RUNTIME_SECONDS="$STACK_RUNTIME_SECONDS" \
    docker compose up --build --abort-on-container-exit --exit-code-from main
  )

  OUTPUT_FILE="$RESULTS_DIR/results_producer_batch_${BATCH_SIZE}.csv"
  if [[ -f "$BASE_RESULTS_FILE" ]]; then
    mv "$BASE_RESULTS_FILE" "$OUTPUT_FILE"
    echo "Saved: $OUTPUT_FILE"
  else
    echo "Warning: expected results file not found at $BASE_RESULTS_FILE"
  fi

  (
    cd "$ROOT_DIR"
    docker compose down --remove-orphans >/dev/null 2>&1 || true
  )
done

echo "All runs finished. Results are in: $RESULTS_DIR"