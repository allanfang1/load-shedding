# Modelling Module — Technical Documentation

This document explains the design, runtime flow, extension points, and operational usage of the `src/modelling` module.

---

## 1) Purpose and Scope

The modelling module predicts the runtime of graph algorithms before they execute in the online load-shedding pipeline.

It provides:

1. **Data collection**: run graph algorithms over representative graphs and record observed runtimes.
2. **Feature engineering**: extract lightweight graph-level features in constant time.
3. **Model training**: fit a regressor (currently Random Forest) per algorithm.
4. **Inference**: predict runtime of a target algorithm on a new graph.

The implementation is intended for low-latency use in streaming environments where prediction overhead must remain negligible compared to algorithm execution.

---

## 2) Module Layout

`src/modelling/`

- `main.py` — CLI orchestration (`collect`, `train`, `predict`, `run-all`).
- `algorithms.py` — algorithm implementations + registry abstraction.
- `feature_extraction.py` — graph feature extraction and feature-vector ordering.
- `runtime_predictor.py` — train/predict/persist ML runtime model.
- `requirements.txt` — Python package dependencies for this module.
- `models/` — default persisted artifacts (`model.joblib`, `scaler.joblib`, `meta.json`).

---

## 3) End-to-End Architecture

### 3.1 High-level flow

1. **Graph source**:
   - From static edge-list files (`--graph-dir`), and/or
   - From sampled window snapshots over a real stream file (`--sample-file`).
2. **Feature extraction**: compute fixed feature vector from each graph.
3. **Timing measurement**: run selected algorithm(s), measure median wall-clock runtime.
4. **Dataset assembly**: write rows to CSV.
5. **Model fit**: train regressor on feature matrix `X` and runtime target `y`.
6. **Serialization**: persist model/scaler/metadata.
7. **Prediction path**: load artifacts, extract features from new graph, infer runtime.

### 3.2 Data contracts

**Collected row schema** (`timings.csv`):

- `num_nodes` (float)
- `num_edges` (float)
- `density` (float)
- `avg_degree` (float)
- `is_directed` (float/int)
- `graph_label` (string)
- `algorithm` (string)
- `runtime` (float, seconds; can be `NaN` on failures)

**Persisted model artifacts** (`model-dir`):

- `model.joblib` — fitted `RandomForestRegressor`
- `scaler.joblib` — fitted `StandardScaler`
- `meta.json` — metadata:
  - `algorithm_name`
  - `feature_names`

---

## 4) Detailed File-by-File Behavior

## 4.1 `algorithms.py`

Defines callable graph algorithms and exposes a registry interface.

### Current registry entries

- `betweenness_centrality`
- `approx_betweenness_centrality`
- `pagerank`

Disabled/commented examples:

- `closeness_centrality`
- `clustering_coefficient`

### Public registry API

- `list_algorithms() -> list[str]`
- `get_algorithm(name: str) -> callable`

### Extension contract

Algorithm functions should:

- Accept a NetworkX graph.
- Return computed results (dict/list/etc.).
- Avoid side effects like prints or file I/O.
- Let caller handle timing externally.

This keeps timing measurements focused on algorithm compute rather than logging overhead.

---

## 4.2 `feature_extraction.py`

Provides fixed-order numeric features used for both training and inference.

### Feature design

All features are derived from:

- number of nodes `n`
- number of edges `m`
- directedness flag

Features:

- `num_nodes`
- `num_edges`
- `density`
- `avg_degree`
- `is_directed`

### Complexity model

The module is designed so feature extraction remains effectively constant overhead relative to graph size from the caller perspective (using NetworkX count queries and arithmetic derivations only).

### Deterministic ordering

- `FEATURE_NAMES` defines canonical order.
- `features_to_vector()` uses `FEATURE_NAMES` to produce stable vectors.

This ordering is critical for model consistency between train and predict phases.

---

## 4.3 `runtime_predictor.py`

Encapsulates model lifecycle in class `RuntimePredictor`.

### Model choices

- Regressor: `RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)`
- Preprocessing: `StandardScaler`

### Why this setup

- Random forests capture non-linear scaling patterns better than simple linear models.
- Inference is fast enough for online gating decisions.
- Minimal tuning burden and robust behavior on modest tabular datasets.

### Main methods

- `fit(X, y, algorithm_name="unknown", cv_folds=0)`
  - Fits scaler and model.
  - Computes in-sample `mae` and `r2`.
  - Optional CV MAE metrics when `cv_folds > 0` and enough samples exist.
- `predict(features_dict)`
  - Dict → ordered vector → scaled vector → scalar runtime prediction.
- `predict_batch(X)`
  - Batch inference on matrix input.
- `feature_importances()`
  - Returns per-feature importance map from fitted forest.
- `save(directory)` / `load(directory)`
  - Persists and reconstructs full predictor state.

### Runtime safeguards

`predict`, `predict_batch`, and `feature_importances` fail fast with `RuntimeError` if used prior to fitting/loading.

---

## 4.4 `main.py` (CLI entry point)

Implements four subcommands:

### `collect`

Builds graph list, extracts features, times algorithm(s), writes CSV.

Input graph modes:

- `--graph-dir` for static `.txt` edge-list graphs.
- `--sample-file` for stream snapshot sampling.

Snapshot sampling (`sample_window_snapshots`) behavior:

- Reads all edges from stream file.
- Computes geometric progression of window sizes (`numpy.geomspace`) between a minimum and `max_edges` cap.
- Builds centered sliding windows and creates one graph per window.

Timing behavior:

- `time.perf_counter()` for wall-clock timing.
- Repeats each `(graph, algorithm)` run and stores median (`numpy.median`).

### `train`

- Loads CSV rows.
- Filters rows to `--algo`.
- Converts feature dicts into `X` matrix.
- Fits predictor and prints MAE/R² (+ optional CV if enabled in code).
- Prints feature importances.
- Saves artifacts into `--model-dir`.

### `predict`

- Loads persisted model.
- Loads target graph.
- Extracts features.
- Predicts runtime and prints graph stats + prediction latency.

### `run-all`

Pipeline shortcut: `collect` then `train`.

---

## 5) Input and Output Formats

## 5.1 Edge-list input format

Expected line format:

`src dst [weight] [timestamp]`

- First two columns are required.
- Additional columns are optional and ignored except `weight` in `load_graph_from_edgelist` when present in third column.
- Parsing is whitespace-based.

## 5.2 CSV behavior notes

- Numeric fields are re-cast to floats when loading.
- Failed algorithm runs are currently recorded with `runtime = NaN`.

If NaNs appear in the selected training set, training may fail depending on estimator constraints; pre-cleaning/filtering can be added as a hardening step.

---

## 6) Practical Usage Guide

Assuming working directory is `src/modelling`.

### 6.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 6.2 Collect timing data

```bash
python main.py collect --sample-file ../../data/higgs-activity_time_postprocess.txt --algo pagerank --out timings.csv
```

### 6.3 Train a model

```bash
python main.py train --csv timings.csv --algo pagerank --model-dir models
```

### 6.4 Predict runtime on a new graph

```bash
python main.py predict --model-dir models --graph ../../data/test_graph.txt
```

### 6.5 End-to-end in one command

```bash
python main.py run-all --sample-file ../../data/higgs-activity_time_postprocess.txt --algo pagerank --model-dir models
```

---

## 7) How to Extend the Module

## 7.1 Add a new algorithm target

1. Implement function in `algorithms.py` with signature `fn(G) -> result`.
2. Register it in `ALGORITHM_REGISTRY` with a unique name.
3. Run `collect` for that algorithm.
4. Train a dedicated model with `train --algo <new_name>`.

Notes:

- Keep algorithm deterministic when possible.
- Avoid prints/logging inside algorithm function to reduce timing noise.

## 7.2 Add or modify features

1. Update feature derivation in `feature_extraction.py`.
2. Ensure `FEATURE_NAMES` ordering remains explicit and stable.
3. Re-collect dataset (old CSVs may become schema-incompatible).
4. Re-train models and overwrite or version artifact directories.

Backward compatibility warning: existing `model.joblib`/`scaler.joblib` may be invalid if feature order or feature set changes.

## 7.3 Change model type or hyperparameters

Update `RuntimePredictor.__init__` and potentially evaluation/reporting logic in `fit`.

Recommended process:

1. Keep the same `fit/predict/save/load` public API.
2. Add metrics needed to compare old vs new model.
3. Evaluate on held-out data or cross-validation.
4. Version model directories (`models_v2`, etc.) during transition.

---

## 8) Operational Considerations

## 8.1 Reproducibility

- Random forest is seeded (`random_state=42`) for stable training behavior.
- Data generation remains workload-dependent; keep sample configuration documented.

## 8.2 Performance characteristics

- Data collection is the expensive stage (actual algorithm runs).
- Training is moderate cost and parallelized across CPU cores by scikit-learn.
- Prediction is low-latency (feature extraction + scaler + forest traversal).

## 8.3 Failure modes

- Empty or malformed graph file → graph load errors / no edges.
- Unsupported `--algo` value → registry `KeyError`.
- No matching algorithm rows during `train` → process exits.
- Missing artifacts in `model-dir` during `predict` → file load failure.

---

## 9) Troubleshooting Checklist

1. **Import issues (`modelling.*`)**
   - Run from an environment where `src` is import-resolvable (for example running module from package-aware context).
2. **No graphs discovered**
   - Verify `--graph-dir` path and `.txt` extension.
3. **Training errors from invalid runtime values**
   - Inspect CSV for `NaN` runtime rows and remove/filter before training.
4. **Poor prediction accuracy**
   - Increase graph diversity, snapshot count, and include larger windows representative of production load.
5. **Long collection times**
   - Use `--algo` for one algorithm at a time, reduce `--num-snapshots`, or cap `--max-edges`.

---

## 10) Suggested Contributor Workflow

1. Create or activate a dedicated Python environment.
2. Install `requirements.txt`.
3. Run small `collect` with one algorithm and low snapshot count.
4. Train and inspect feature importances.
5. Validate `predict` on held-out graphs.
6. Iterate on feature set/model configuration.
7. Commit code + updated artifacts/documentation as a coherent unit.

---

## 11) Maintenance and Versioning Recommendations

- Treat model artifacts as build outputs tied to code and feature schema.
- Keep timing datasets versioned (or checksum-tracked) for reproducibility.
- Introduce explicit schema version in `meta.json` if feature set evolves.
- Add automated validation checks for:
  - non-empty datasets,
  - no-NaN training targets,
  - artifact loadability,
  - feature name alignment between code and metadata.

---

## 12) Quick Reference

- Main CLI: `python main.py <collect|train|predict|run-all> ...`
- Algorithms available: `list_algorithms()` in `algorithms.py`
- Feature order contract: `FEATURE_NAMES` in `feature_extraction.py`
- Model persistence entrypoint: `RuntimePredictor.save/load`

This module is designed to remain simple, inspectable, and easy to extend while enabling runtime-aware decision-making in the larger load-shedding system.
