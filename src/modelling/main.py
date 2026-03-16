"""
CLI entry point for the graph-runtime prediction pipeline.

Workflow
-------
1. **collect**  - Load graph files, run algorithms, measure time, save CSV.
2. **train**    - Load CSV, train an ML model per algorithm, save model.
3. **predict**  - Load a saved model + a graph, print predicted runtime.
4. **run-all**  - Do collect → train end-to-end.

Usage examples
--------------
    # Collect timing data from all .txt graphs in ../data/
    python main.py collect --graph-dir ../data --out timings.csv

    # Collect by sampling sliding-window snapshots from a real edge stream
    python main.py collect --sample-file ../data/higgs-activity_time_postprocess.txt --algo pagerank --out timings.csv

    # Train a model for pagerank
    python main.py train --csv timings.csv --algo pagerank --model-dir models

    # Predict runtime for a new graph
    python main.py predict --model-dir models --graph ../data/test_graph.txt

    # End-to-end (sample from real data → train)
    python main.py run-all --sample-file ../data/higgs-activity_time_postprocess.txt --algo pagerank --model-dir models
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

try:
    from modelling.algorithms import ALGORITHM_REGISTRY, get_algorithm, list_algorithms
    from modelling.feature_extraction import (
        FEATURE_NAMES,
        extract_features,
        features_to_vector,
    )
    from modelling.runtime_predictor import RuntimePredictor
except ModuleNotFoundError:
    from algorithms import ALGORITHM_REGISTRY, get_algorithm, list_algorithms
    from feature_extraction import (
        FEATURE_NAMES,
        extract_features,
        features_to_vector,
    )
    from runtime_predictor import RuntimePredictor


# ======================================================================
# Graph I/O helpers
# ======================================================================

def load_graph_from_edgelist(filepath: str, directed: bool = True) -> nx.Graph:
    """Load a graph from a space-separated edge list file.

    Expected format per line:  src dst [weight] [timestamp]
    Columns beyond the first two are optional.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            weight = float(parts[2]) if len(parts) >= 3 else 1.0
            G.add_edge(src, dst, weight=weight)
    return G


def discover_graph_files(directory: str, extensions: tuple = (".txt",)) -> list[str]:
    """Return sorted list of graph files in *directory*."""
    paths = []
    for fname in os.listdir(directory):
        if any(fname.endswith(ext) for ext in extensions):
            full = os.path.join(directory, fname)
            if os.path.isfile(full):
                paths.append(full)
    return sorted(paths)


def sample_window_snapshots(
    filepath: str,
    num_snapshots: int = 30,
    max_edges: int = 5000,
    directed: bool = True,
) -> list[tuple[str, nx.Graph]]:
    """Sample sliding-window-style graph snapshots from a real edge-list file.

    Reads *all* edges once, then takes ``num_snapshots`` windows of varying
    sizes spread across the stream.  This produces training graphs whose
    structure (degree distribution, density, clustering) mirrors the actual
    workload seen by the streaming system.

    Parameters
    ----------
    filepath : str
        Path to a space-separated edge-list file (src dst [weight] [timestamp]).
    num_snapshots : int
        How many sub-graphs to produce.
    max_edges : int
        Cap window size to avoid prohibitively long algorithm runs.
    directed : bool
        Whether to build directed graphs.

    Returns
    -------
    list of (label, nx.Graph) tuples.
    """
    # 1. Read all edges
    edges: list[tuple[str, str]] = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))

    total = len(edges)
    if total == 0:
        raise ValueError(f"No edges found in {filepath}")

    fname = Path(filepath).stem

    # 2. Pick window boundaries
    #    We want varied graph sizes — log-spaced from a small window up to
    #    max_edges (or total if smaller), so we get both tiny and large snapshots.
    min_edges = max(10, max_edges // 200)
    upper = min(total, max_edges)
    import numpy as _np
    window_sizes = sorted(set(
        int(s) for s in _np.geomspace(min_edges, upper, num=num_snapshots)
    ))

    # 3. Build a graph for each window
    snapshots: list[tuple[str, nx.Graph]] = []
    for ws in window_sizes:
        # Slide the window to the middle of the stream (or end if too large)
        start = max(0, (total - ws) // 2)
        G = nx.DiGraph() if directed else nx.Graph()
        for src, dst in edges[start : start + ws]:
            G.add_edge(src, dst)
        label = f"{fname}_win{ws}_start{start}"
        snapshots.append((label, G))
        print(f"  Snapshot {label}: n={G.number_of_nodes()}, m={G.number_of_edges()}")

    return snapshots


# ======================================================================
# Data collection
# ======================================================================

def time_algorithm(algo_fn, G: nx.Graph, repeats: int = 3) -> float:
    """Run *algo_fn* on *G* ``repeats`` times and return **median** wall time."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        algo_fn(G)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return float(np.median(times))


def collect_timings(
    graphs: list[tuple[str, nx.Graph]],
    algo_names: list[str] | None = None,
    repeats: int = 3,
) -> list[dict]:
    """Run every registered algorithm on every graph; return rows of data.

    Each row is a dict:  {**features, 'graph_label': ..., 'algorithm': ..., 'runtime': ...}
    """
    if algo_names is None:
        algo_names = list_algorithms()

    rows = []
    total = len(graphs) * len(algo_names)
    done = 0
    for label, G in graphs:
        features = extract_features(G)
        for algo_name in algo_names:
            done += 1
            algo_fn = get_algorithm(algo_name)
            print(f"  [{done}/{total}] {algo_name} on {label} "
                  f"(n={G.number_of_nodes()}, m={G.number_of_edges()}) ...",
                  end=" ", flush=True)
            try:
                rt = time_algorithm(algo_fn, G, repeats=repeats)
                print(f"{rt:.4f}s")
            except Exception as e:
                print(f"FAILED ({e})")
                rt = float("nan")
            row = {**features, "graph_label": label, "algorithm": algo_name, "runtime": rt}
            rows.append(row)
    return rows


def save_timings_csv(rows: list[dict], path: str) -> None:
    """Write collected rows to a CSV file."""
    if not rows:
        print("No data to save.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


def load_timings_csv(path: str) -> list[dict]:
    """Load timing rows from CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert numeric fields back from strings
    for row in rows:
        for key in FEATURE_NAMES + ["runtime"]:
            if key in row:
                row[key] = float(row[key])
    return rows


# ======================================================================
# CLI commands
# ======================================================================

def cmd_collect(args):
    """Collect timing data from graph files and/or synthetic graphs."""
    graphs: list[tuple[str, nx.Graph]] = []

    # Load from directory
    if args.graph_dir:
        files = discover_graph_files(args.graph_dir)
        if not files:
            print(f"No graph files found in {args.graph_dir}")
        for fp in files:
            label = Path(fp).stem
            print(f"Loading {label} ...")
            try:
                G = load_graph_from_edgelist(fp, directed=True)
                graphs.append((label, G))
            except Exception as e:
                print(f"  Skipped {fp}: {e}")

    # Sample window snapshots from a real edge stream
    if args.sample_file:
        print(f"Sampling window snapshots from {args.sample_file} ...")
        graphs.extend(sample_window_snapshots(
            args.sample_file,
            num_snapshots=args.num_snapshots,
            max_edges=args.max_edges,
        ))

    if not graphs:
        print("No graphs to process.  Use --graph-dir and/or --sample-file.")
        sys.exit(1)

    algo_names = [args.algo] if args.algo else None
    print(f"\nCollecting timings (repeats={args.repeats}) ...")
    rows = collect_timings(graphs, algo_names=algo_names, repeats=args.repeats)
    save_timings_csv(rows, args.out)


def cmd_train(args):
    """Train an ML model from a collected CSV."""
    rows = load_timings_csv(args.csv)

    # Filter to the requested algorithm
    if args.algo:
        rows = [r for r in rows if r["algorithm"] == args.algo]
    if not rows:
        print(f"No data for algorithm '{args.algo}' in {args.csv}")
        sys.exit(1)

    algo_name = rows[0]["algorithm"]
    X = np.array([features_to_vector(r) for r in rows])
    y = np.array([r["runtime"] for r in rows])

    print(f"Training on {len(y)} samples for '{algo_name}' ...")
    predictor = RuntimePredictor()
    metrics = predictor.fit(X, y, algorithm_name=algo_name)

    print(f"  MAE:        {metrics['mae']:.6f}s")
    print(f"  R²:         {metrics['r2']:.4f}")
    if "cv_mean_mae" in metrics:
        print(f"  CV MAE:     {metrics['cv_mean_mae']:.6f} ± {metrics['cv_std_mae']:.6f}")

    importances = predictor.feature_importances()
    print("\n  Feature importances:")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}  {imp:.4f}")

    predictor.save(args.model_dir)
    print(f"\nModel saved to {args.model_dir}/")


def cmd_predict(args):
    """Load a trained model and predict runtime for a graph."""
    predictor = RuntimePredictor.load(args.model_dir)
    G = load_graph_from_edgelist(args.graph, directed=True)
    features = extract_features(G)

    start = time.perf_counter()
    predicted = predictor.predict(features)
    pred_time = time.perf_counter() - start

    print(f"Graph:       {args.graph}")
    print(f"  Nodes:     {features['num_nodes']}")
    print(f"  Edges:     {features['num_edges']}")
    print(f"  Density:   {features['density']:.6f}")
    print(f"  Avg degree:{features['avg_degree']:.2f}")
    print(f"\nPredicted runtime for '{predictor.algorithm_name}': {predicted:.6f}s")
    print(f"(prediction itself took {pred_time*1e6:.0f}µs)")


def cmd_run_all(args):
    """End-to-end: collect → train."""
    csv_path = args.out or os.path.join(args.model_dir, "timings.csv")

    # Collect
    args.out = csv_path
    cmd_collect(args)

    # Train
    args.csv = csv_path
    cmd_train(args)


# ======================================================================
# Argument parser
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph algorithm runtime predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- collect -------------------------------------------------------
    p_collect = sub.add_parser("collect", help="Run algorithms and collect timing data")
    p_collect.add_argument("--graph-dir", type=str, default=None,
                           help="Directory with graph edge-list files (.txt)")
    p_collect.add_argument("--sample-file", type=str, default=None,
                           help="Edge-list file to sample window snapshots from (real workload)")
    p_collect.add_argument("--num-snapshots", type=int, default=15,
                           help="Number of window snapshots to sample (default: 15)")
    p_collect.add_argument("--max-edges", type=int, default=5000,
                           help="Cap snapshot size in edges (default: 5000)")
    p_collect.add_argument("--algo", type=str, default=None,
                           help=f"Single algorithm to run (default: all). "
                                f"Choices: {list_algorithms()}")
    p_collect.add_argument("--repeats", type=int, default=1,
                           help="Timing repetitions per (graph, algo) pair")
    p_collect.add_argument("--out", type=str, default="timings.csv",
                           help="Output CSV path")

    # -- train ---------------------------------------------------------
    p_train = sub.add_parser("train", help="Train an ML model from collected CSV")
    p_train.add_argument("--csv", type=str, required=True, help="Timing CSV file")
    p_train.add_argument("--algo", type=str, required=True,
                         help="Algorithm name to train a model for")
    p_train.add_argument("--model-dir", type=str, default="models",
                         help="Directory to save the trained model")

    # -- predict -------------------------------------------------------
    p_predict = sub.add_parser("predict", help="Predict runtime for a graph")
    p_predict.add_argument("--model-dir", type=str, required=True,
                           help="Directory with saved model")
    p_predict.add_argument("--graph", type=str, required=True,
                           help="Path to graph edge-list file")

    # -- run-all -------------------------------------------------------
    p_all = sub.add_parser("run-all", help="Collect + Train end-to-end")
    p_all.add_argument("--graph-dir", type=str, default=None)
    p_all.add_argument("--sample-file", type=str, default=None,
                       help="Edge-list file to sample window snapshots from")
    p_all.add_argument("--num-snapshots", type=int, default=15,
                       help="Number of window snapshots to sample")
    p_all.add_argument("--max-edges", type=int, default=5000,
                       help="Cap snapshot size in edges")
    p_all.add_argument("--algo", type=str, required=True)
    p_all.add_argument("--repeats", type=int, default=1)
    p_all.add_argument("--model-dir", type=str, default="models")
    p_all.add_argument("--out", type=str, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cmd_map = {
        "collect": cmd_collect,
        "train": cmd_train,
        "predict": cmd_predict,
        "run-all": cmd_run_all,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()