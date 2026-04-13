"""
CLI entry point for the graph-runtime prediction pipeline.

Workflow
-------
1. **collect**  - Load graph files, run algorithms, measure time, save CSV.
2. **train**    - Load CSV, train an ML model per algorithm, save model.
3. **run-all**  - Do collect → train end-to-end.

Usage examples
--------------
    # Collect timing data from all .txt graphs in ../data/
    python main.py collect --graph-dir ../data --out timings.csv

    # Collect by sampling sliding-window snapshots from a real edge stream
    python main.py collect --sample-file ../data/higgs-activity_time_postprocess.txt --algo pagerank --out timings.csv

    # Train a model for pagerank
    python main.py train --csv timings.csv --algo pagerank --model-dir models

    # End-to-end (sample from real data → train)
    python main.py run-all --sample-file ../data/higgs-activity_time_postprocess.txt --algo pagerank --model-dir models
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator

import networkx as nx
import numpy as np
from sklearn.utils import shuffle

try:
    from modelling_s_v2.algorithms import ALGORITHM_REGISTRY, get_algorithm, list_algorithms
    from modelling_s_v2.mock_window_manager import MockWindowManager
    from modelling_s_v2.feature_extraction import (
        FEATURE_NAMES,
    )
    from modelling_s_v2.runtime_predictor import RuntimePredictor
except ModuleNotFoundError:
    from algorithms import ALGORITHM_REGISTRY, get_algorithm, list_algorithms
    from mock_window_manager import MockWindowManager
    from feature_extraction import (
        FEATURE_NAMES,
    )
    from runtime_predictor import RuntimePredictor


# ======================================================================
# Graph I/O helpers
# ======================================================================

# Per (snapshot, s): run this many random ingest-fraction variants.
RANDOM_INGEST_FRACTION_SAMPLES = 12


def build_staged_window_manager(
    edges: list[tuple[str, str]],
    directed: bool = True,
) -> MockWindowManager:
    """Create a window manager with all edges staged (not in graph yet)."""
    graph = nx.DiGraph() if directed else nx.Graph()
    wm = MockWindowManager(graph, None)
    for src, dst in edges:
        wm.stageEdge(src, dst, t=0)
    return wm

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


def load_window_manager_from_edgelist(filepath: str, directed: bool = True) -> MockWindowManager:
    """Load an edge-list directly into a ``MockWindowManager``."""
    edges: list[tuple[str, str]] = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            edges.append((src, dst))
    return build_staged_window_manager(edges, directed=directed)


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
    max_edges: int = 10,
    directed: bool = True,
) -> Iterator[tuple[str, MockWindowManager]]:
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

    Yields
    ------
    (label, MockWindowManager) tuples.
    """
    if max_edges <= 0:
        raise ValueError("max_edges must be > 0")
    if num_snapshots <= 0:
        raise ValueError("num_snapshots must be > 0")

    def iter_valid_edges() -> Iterator[tuple[str, str]]:
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                if len(parts) >= 3 and parts[2] == "-1":
                    continue
                yield parts[0], parts[1]

    # 1) First pass: count usable edges only.
    total = 0
    for _ in iter_valid_edges():
        total += 1

    if total == 0:
        raise ValueError(f"No edges found in {filepath}")

    fname = Path(filepath).stem

    # 2) Choose varied window sizes (log-spaced) and anchor positions near stream start.
    min_edges = 100000
    upper = min(total, max_edges)
    window_sizes = sorted(set(int(s) for s in np.linspace(min_edges, upper, num=num_snapshots)))
    # window_sizes = sorted(set(int(s) for s in np.geomspace(min_edges, upper, num=num_snapshots)))


    # Build (window_size, end_index, position_idx) specs where end_index is
    # 1-based in the full stream. For each window size, emit up to 3 contiguous
    # windows from the beginning of the stream:
    #   [1..ws], [ws+1..2ws], [2ws+1..3ws]
    # (clamped by stream length).
    positions_per_size = 3
    specs: list[tuple[int, int, int]] = []
    for ws in window_sizes:
        end_candidates = [
            min(ws * pos_idx, total)
            for pos_idx in range(1, positions_per_size + 1)
        ]
        deduped_end_indices = []
        seen = set()
        for end_idx in end_candidates:
            end_idx = max(ws, min(end_idx, total))
            if end_idx in seen:
                continue
            seen.add(end_idx)
            deduped_end_indices.append(end_idx)
        for pos_idx, end_idx in enumerate(deduped_end_indices, start=1):
            specs.append((ws, end_idx, pos_idx))

    # 3) Second pass: maintain bounded sliding buffer and emit snapshots at targets.
    edge_buffer: deque[tuple[str, str]] = deque(maxlen=max_edges)
    current_edge_idx = 0
    spec_idx = 0

    for src, dst in iter_valid_edges():
        current_edge_idx += 1
        edge_buffer.append((src, dst))

        while spec_idx < len(specs):
            ws, end_idx, pos_idx = specs[spec_idx]
            if current_edge_idx < end_idx:
                break

            if len(edge_buffer) < ws:
                spec_idx += 1
                continue

            snapshot_edges = list(edge_buffer)[-ws:]
            wm = build_staged_window_manager(snapshot_edges, directed=directed)

            start = max(0, end_idx - ws)
            label = f"{fname}_win{ws}_p{pos_idx}_start{start}"
            print(f"  Snapshot {label}: n={len(wm.degree_count)}, m={len(wm.edge_count)}")
            yield label, wm
            spec_idx += 1

        if spec_idx >= len(specs):
            break


def iter_graph_window_managers(
    graph_dir: str | None,
    sample_file: str | None,
    num_snapshots: int,
    max_edges: int,
) -> Iterator[tuple[str, MockWindowManager]]:
    """Yield graph windows one at a time for bounded-memory collection."""
    if graph_dir:
        files = discover_graph_files(graph_dir)
        if not files:
            print(f"No graph files found in {graph_dir}")
        for fp in files:
            label = Path(fp).stem
            print(f"Loading {label} ...")
            try:
                wm = load_window_manager_from_edgelist(fp, directed=True)
                yield label, wm
            except Exception as e:
                print(f"  Skipped {fp}: {e}")

    if sample_file:
        print(f"Sampling window snapshots from {sample_file} ...")
        yield from sample_window_snapshots(
            sample_file,
            num_snapshots=num_snapshots,
            max_edges=max_edges,
        )


# ======================================================================
# Data collection
# ======================================================================

def time_algorithm(algo_fn, G: nx.Graph, repeats: int = 3) -> tuple[float, object | None]:
    """Run *algo_fn* on *G* ``repeats`` times.

    Returns
    -------
    tuple
        (median_wall_time_seconds, first_run_result)
    """
    times = []
    first_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = algo_fn(G)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if first_result is None:
            first_result = result
    return float(np.median(times)), first_result


def top_k_from_mapping(result: object, k: int = 5) -> list[tuple[str, float]]:
    """Extract top-k (key, value) pairs from a mapping-like algorithm result."""
    if not isinstance(result, dict):
        return []

    scored_items: list[tuple[str, float]] = []
    for key, value in result.items():
        try:
            scored_items.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue

    scored_items.sort(key=lambda x: x[1], reverse=True)
    return scored_items[:k]


def clone_window_manager(wm: MockWindowManager) -> MockWindowManager:
    """Create a safe copy of a MockWindowManager without using deepcopy."""
    graph_copy = wm.graph.copy()
    cloned = MockWindowManager(graph_copy, wm.algo, wm.base_time)
    cloned.in_moments.s1 = wm.in_moments.s1
    cloned.in_moments.s2 = wm.in_moments.s2
    cloned.in_moments.s3 = wm.in_moments.s3
    cloned.out_moments.s1 = wm.out_moments.s1
    cloned.out_moments.s2 = wm.out_moments.s2
    cloned.out_moments.s3 = wm.out_moments.s3

    curr = wm.timed_list.head
    while curr is not None:
        cloned.timed_list.append(curr.src, curr.dst, curr.t)
        cloned.edge_count[(curr.src, curr.dst)] += 1
        curr = curr.next

    for node, degree in wm.degree_count.items():
        cloned.degree_count[node] = degree

    return cloned


def build_window_state_features(
    wm: MockWindowManager,
    prefix: str,
    percent_incoming: float | None = None,
) -> dict[str, float]:
    """Build features from WindowManager-like state, not from materialized graph."""
    n = len(wm.degree_count)
    m = len(wm.edge_count)

    features = {
        f"{prefix}_num_nodes": float(n),
        f"{prefix}_num_edges": float(m),
        f"{prefix}_log_num_nodes": float(np.log2(n)) if n > 0 else 0.0,
        f"{prefix}_log_num_edges": float(np.log2(m)) if m > 0 else 0.0,
        f"{prefix}_avg": wm.in_moments.get_mean(n),
        f"{prefix}_var_in": wm.in_moments.get_variance(n),
        f"{prefix}_var_out": wm.out_moments.get_variance(n),
        f"{prefix}_skew_in": wm.in_moments.get_skewness(n),
        f"{prefix}_skew_out": wm.out_moments.get_skewness(n),
    }
    if prefix == "pre":
        features["percent_incoming"] = float(percent_incoming or 0.0)
    return features


def collect_timings(
    graphs: Iterable[tuple[str, MockWindowManager]],
    algo_names: list[str] | None = None,
    repeats: int = 3,
) -> Iterator[dict]:
    """Run every registered algorithm on every graph; return rows of data.

    Each row is a dict:  {**features, 'graph_label': ..., 'algorithm': ..., 'runtime': ...}
    """
    if algo_names is None:
        algo_names = list_algorithms()

    num_s_samples = 16
    low, high = 0.01, 6
    done = 0

    for label, wm in graphs:
        for s_value in np.sort(np.exp(np.random.uniform(np.log(low), np.log(high), size=num_s_samples))):
            ingest_fractions = np.sort(np.random.uniform(0.0, 1.0, size=RANDOM_INGEST_FRACTION_SAMPLES).tolist())

            any_topology_change = False
            for sample_idx, ingest_fraction in enumerate(ingest_fractions):
                tmp_wm = clone_window_manager(wm)

                total_staged = tmp_wm.timed_list.size
                ingest_count = int(round(float(ingest_fraction) * total_staged))
                ingest_count = max(0, min(ingest_count, total_staged))
                first_pending_node = tmp_wm.materialize_prefix(ingest_count)

                percent_incoming = (
                    ((total_staged - ingest_count) / total_staged) if total_staged > 0 else 0.0
                )

                pre_features = build_window_state_features(
                    tmp_wm,
                    prefix="pre",
                    percent_incoming=percent_incoming,
                )

                begin_shed = time.perf_counter()
                tmp_wm.modifiedSpectralSparsity(s_value, start_node=first_pending_node)
                end_shed = time.perf_counter()
                post_features = build_window_state_features(tmp_wm, prefix="post")

                topology_changed = (
                    pre_features["pre_num_nodes"] != post_features["post_num_nodes"]
                    or pre_features["pre_num_edges"] != post_features["post_num_edges"]
                )
                any_topology_change = any_topology_change or topology_changed

                for algo_name in algo_names:
                    done += 1
                    algo_fn = get_algorithm(algo_name)
                    print(f"  [{done}] {algo_name} on {label} with s={s_value:.2f}, ingest={ingest_fraction:.3f} "
                        f"(n={tmp_wm.graph.number_of_nodes()}, m={tmp_wm.graph.number_of_edges()}) ...",
                        end=" ", flush=True)
                    try:
                        rt, algo_result = time_algorithm(algo_fn, tmp_wm.graph, repeats=repeats)
                        print(f"{rt:.4f}s")
                    except Exception as e:
                        print(f"FAILED ({e})")
                        rt = float("nan")
                        algo_result = None

                    pagerank_top10 = ""
                    if algo_name == "pagerank":
                        pagerank_top10 = json.dumps(top_k_from_mapping(algo_result, k=10))

                    row = {
                        **pre_features,
                        **post_features,
                        "graph_label": label,
                        "s_value": s_value,
                        "ingest_fraction": float(ingest_fraction),
                        "algorithm": algo_name,
                        "runtime": rt,
                        "shed_time": end_shed - begin_shed,
                        "budget": max(0.0, rt + (end_shed - begin_shed)),
                        "pagerank_top10": pagerank_top10,

                    }
                    yield row

            if not any_topology_change:
                print(
                    f"  -> No topology change after sparsification for {label} at s={s_value:.2f} "
                    "across ingest-fraction samples; skipping remaining s-values for this graph."
                )
                break


def collect_ground_truth_timings(
    graphs: Iterable[tuple[str, MockWindowManager]],
    algo_names: list[str] | None = None,
) -> Iterator[dict]:
    """Collect no-shedding runtime rows as a streaming iterator."""
    gt_algo_names = algo_names if algo_names is not None else list_algorithms()
    gt_progress = 0

    for label, wm in graphs:
        wm_full = clone_window_manager(wm)
        wm_full.materialize_prefix(wm_full.timed_list.size)

        features = build_window_state_features(
            wm_full,
            prefix="pre",
            percent_incoming=0.0,
        )
        for algo_name in gt_algo_names:
            gt_progress += 1
            algo_fn = get_algorithm(algo_name)
            print(
                f"  [GT {gt_progress}] {algo_name} on {label} "
                f"(n={wm_full.graph.number_of_nodes()}, m={wm_full.graph.number_of_edges()}) ...",
                end=" ",
                flush=True,
            )
            try:
                begin = time.perf_counter()
                algo_result = algo_fn(wm_full.graph)
                rt = time.perf_counter() - begin
                print(f"{rt:.4f}s")
            except Exception as e:
                print(f"FAILED ({e})")
                rt = float("nan")
                algo_result = None

            pagerank_top10 = ""
            if algo_name == "pagerank":
                pagerank_top10 = json.dumps(top_k_from_mapping(algo_result, k=10))

            yield {
                **features,
                "graph_label": label,
                "algorithm": algo_name,
                "runtime": rt,
                "budget": max(0.0, rt),
                "pagerank_top10": pagerank_top10,
            }


def save_timings_csv(rows: Iterable[dict], path: str) -> int:
    """Write collected rows to a CSV file (streaming)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    row_iter = iter(rows)
    first_row = next(row_iter, None)
    if first_row is None:
        print("No data to save.")
        return 0

    fieldnames = list(first_row.keys())
    count = 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first_row)
        count += 1
        for row in row_iter:
            writer.writerow(row)
            count += 1
    print(f"Saved {count} rows to {path}")
    return count


def load_timings_csv(path: str) -> list[dict]:
    """Load timing rows from CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert numeric fields back from strings
    for row in rows:
        for key in FEATURE_NAMES: # TODO automate loading target
            if key in row:
                row[key] = float(row[key])
    return rows


def parse_feature_list(raw: str | None, available: list[str]) -> list[str]:
    """Parse a comma-separated feature list.

    Accepted values
    ---------------
    - "a,b,c" -> explicit ordered subset
    """
    if raw is None:
        raise ValueError(
            "No features provided. Pass --features as a comma-separated list."
        )

    selected = [x.strip() for x in raw.split(",") if x.strip()]
    if not selected:
        raise ValueError("No features selected.")

    unknown = [x for x in selected if x not in available]
    if unknown:
        raise ValueError(
            f"Unknown feature(s): {unknown}. "
            f"Available features: {available}"
        )

    duplicates = [x for x in set(selected) if selected.count(x) > 1]
    if duplicates:
        raise ValueError(f"Duplicate feature(s): {sorted(duplicates)}")

    return selected


# ======================================================================
# CLI commands
# ======================================================================

def cmd_collect(args):
    """Collect timing data from graph files and/or synthetic graphs."""
    has_graph_dir_files = bool(args.graph_dir and discover_graph_files(args.graph_dir))
    has_sample_source = bool(args.sample_file)
    if not has_graph_dir_files and not has_sample_source:
        print("No graphs to process.  Use --graph-dir and/or --sample-file.")
        sys.exit(1)

    out_path_str = args.out or os.path.join("models", "timings.csv")
    args.out = out_path_str

    algo_names = [args.algo] if args.algo else None
    print(f"\nCollecting timings (repeats={args.repeats}) ...")
    rows = collect_timings(
        iter_graph_window_managers(
            graph_dir=args.graph_dir,
            sample_file=args.sample_file,
            num_snapshots=args.num_snapshots,
            max_edges=args.max_edges,
        ),
        algo_names=algo_names,
        repeats=args.repeats,
    )
    collected_count = save_timings_csv(rows, args.out)
    if collected_count == 0:
        print("No timing rows collected.")
        sys.exit(1)

    out_path = Path(args.out)
    gt_out = str(out_path.with_name(f"{out_path.stem}_ground_truth{out_path.suffix or '.csv'}"))
    print("\nCollecting ground-truth timings (no shedding) ...")
    gt_rows = collect_ground_truth_timings(
        iter_graph_window_managers(
            graph_dir=args.graph_dir,
            sample_file=args.sample_file,
            num_snapshots=args.num_snapshots,
            max_edges=args.max_edges,
        ),
        algo_names=algo_names,
    )
    save_timings_csv(gt_rows, gt_out)


def cmd_train(args):
    """Train an ML model from a collected CSV."""
    csv_path = args.csv or os.path.join(args.model_dir, "timings2.csv")

    rows = load_timings_csv(csv_path)

    # Filter to the requested algorithm
    if args.algo:
        rows = [r for r in rows if r["algorithm"] == args.algo]
    if not rows:
        print(f"No data for algorithm '{args.algo}' in {csv_path}")
        sys.exit(1)

    target_name = "s_value"
    excluded_columns = {"graph_label", "algorithm", target_name, "pagerank_top10"}
    available_numeric_features = [
        key for key in rows[0].keys() if key not in excluded_columns
    ]

    try:
        if args.features:
            selected_features = parse_feature_list(args.features, available_numeric_features)
        else:
            selected_features = available_numeric_features
    except ValueError as e:
        print(f"Invalid --features: {e}")
        print(f"Selectable feature columns: {available_numeric_features}")
        sys.exit(1)

    algo_name = rows[0]["algorithm"]
    try:
        X = np.array([
            [float(r[name]) for name in selected_features]
            for r in rows
        ])
    except KeyError as e:
        print(
            f"Feature {e} not present in CSV rows. "
            f"Selected features: {selected_features}"
        )
        sys.exit(1)

    y = np.array([float(r[target_name]) for r in rows])

    X, y = shuffle(X, y, random_state=42)

    print(f"Training on {len(y)} samples for '{algo_name}' ...")
    print(f"Using features (in order): {selected_features}")

    predictor = RuntimePredictor()

    metrics = predictor.fit(
        X,
        y,
        algorithm_name=algo_name,
        cv_folds=args.cv_folds,
        feature_names=selected_features,
    )

    print(f"  MAE:        {metrics['mae']:.6f}")
    print(f"  R²:         {metrics['r2']:.4f}")
    if "oob_r2" in metrics:
        print(f"  OOB R²:     {metrics['oob_r2']:.4f}")
    if "cv_mean_mae" in metrics:
        folds_used = metrics.get("cv_folds_used", args.cv_folds)
        print(f"  CV MAE:     {metrics['cv_mean_mae']:.6f} ± {metrics['cv_std_mae']:.6f}  ({folds_used} folds)")
        if "cv_mae_scores" in metrics:
            fold_text = ", ".join(f"{v:.4f}" for v in metrics["cv_mae_scores"])
            print(f"  Fold MAE:   [{fold_text}]")
        train_cv_gap = metrics["cv_mean_mae"] - metrics["mae"]
        ratio = metrics["cv_mean_mae"] / max(metrics["mae"], 1e-12)
        print(f"  CV-Train gap: {train_cv_gap:+.6f}  (ratio={ratio:.2f}x)")
        if ratio > 1.20:
            print("  Note: gap suggests possible overfitting; consider more diverse training samples")
    elif args.cv_folds > 0:
        print(f"  CV skipped: need at least {args.cv_folds} samples, got {len(y)}")

    importances = predictor.feature_importances()
    print("\n  Feature importances:")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}  {imp:.4f}")

    predictor.save(args.model_dir)
    print(f"\nModel saved to {args.model_dir}/")


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
    p_collect.add_argument("--out", type=str, default=None,
                           help="Output CSV path")

    # -- train ---------------------------------------------------------
    p_train = sub.add_parser("train", help="Train an ML model from collected CSV")
    p_train.add_argument("--csv", type=str, help="Timing CSV file")
    p_train.add_argument("--algo", type=str, required=True,
                         help="Algorithm name to train a model for")
    p_train.add_argument("--features", type=str, default=None,
                         help="Comma-separated feature list to train on (default: all numeric non-target columns)")
    p_train.add_argument("--model-dir", type=str, default="models",
                         help="Directory to save the trained model")
    p_train.add_argument("--cv-folds", type=int, default=5,
                         help="Cross-validation folds (0 to disable, default: 5)")

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
    p_all.add_argument("--features", type=str, default=None,
                       help="Comma-separated feature list to pass to train")
    p_all.add_argument("--cv-folds", type=int, default=5,
                       help="Cross-validation folds used in train (default: 5)")
    p_all.add_argument("--model-dir", type=str, default="models")
    p_all.add_argument("--out", type=str, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    np.random.seed(42)

    cmd_map = {
        "collect": cmd_collect,
        "train": cmd_train,
        "run-all": cmd_run_all,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()