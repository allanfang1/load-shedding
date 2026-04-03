# Modelling Module — Technical Documentation

This document explains the design, runtime flow, extension points, and operational usage of the `src/modelling_s` module.

---

## 1) Purpose and Scope

The modelling module is used to learn and export a model for sparsification control value (`s_value`) estimation in the online load-shedding pipeline.

It provides:

1. **Data collection**: sample representative graph windows, apply sparsification with multiple `s` values, and record timings.
2. **Features**: extract lightweight (constant time) graph-level features before/after sparsification.
3. **Training**: fit a regressor (Random Forest) to predict `s_value`.
4. **Export**: persist trained model artifacts for use by external inference components.

The implementation is intended for low-latency use in streaming environments where prediction overhead must remain negligible compared to algorithm execution.

---

## 2) Module Layout

`src/modelling_s/`

- `main.py` — CLI orchestration (`collect`, `train`, `run-all`).
- `algorithms.py` — algorithm implementations + registry abstraction.
- `feature_extraction.py` — graph feature extraction and vectorization.
- `runtime_predictor.py` — train/predict/persist ML runtime model.
- `mock_window_manager.py` — lightweight window manager used for simulation. Respects input order.
- `requirements.txt` — Python package dependencies for this module.
- `models/` — default persisted artifacts (`model.joblib`, `meta.json`).

---

## 3) End-to-End Architecture

### 3.1 High-level flow

1. **Graph source**:
   - From static edge-list snapshot files (`--graph-dir`), and/or
   - From edge-list file as a stream (`--sample-file`).
2. **Feature extraction**: compute graph feature vectors.
3. **Sparsify + timing**: for each sampled graph, sweep `s_value`, sparsify (timed), run selected algorithm(s) (timed).
4. **Write**: write data rows to CSV.
5. **Model fit**: train regressor on feature matrix `X` and target `y` (`s_value`).
6. **Serialization**: persist model/metadata.
7. **External inference path**: downstream components load artifacts and infer targets outside this module CLI.

### 3.2 Data contracts

**Collected row schema** (`timings.csv`):

- `pre_num_nodes`, `pre_num_edges`, `pre_log_num_nodes`, `pre_log_num_edges`, `pre_is_directed`
- `post_num_nodes`, `post_num_edges`, `post_log_num_nodes`, `post_log_num_edges`, `post_is_directed`
- `graph_label` (string)
- `s_value` (float)
- `algorithm` (string)
- `runtime` (float, seconds; can be `NaN` on failures)
- `shed_time` (float, seconds)
- `budget` (float, currently `runtime + shed_time`)

**Persisted model artifacts** (`model-dir`):

- `model.joblib` — fitted `RandomForestRegressor`
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

Disabled:

- `closeness_centrality`
- `clustering_coefficient`

### API

- `list_algorithms() -> list[str]`
- `get_algorithm(name: str) -> callable`

### Extension contract

Algorithm functions should:

- Accept a NetworkX graph.
- Return computed results (dict/list/etc.).

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
- `log_num_nodes`
- `log_num_edges`
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

- Regressor: `RandomForestRegressor(n_estimators=120, min_samples_leaf=2, min_samples_split=4, max_features="sqrt", oob_score=True, random_state=42, n_jobs=-1)`

### Why this setup

- Random forests capture non-linear and discontinuous patterns better than simple linear models.
- Inference is fast enough for online gating decisions.
- Minimal tuning burden and robust behaviour on modest tabular datasets.

### Main methods

- `fit(X, y, algorithm_name="unknown", cv_folds=0, feature_names=None, sample_weight=None)`
  - Fits model.
  - Computes in-sample `mae` and `r2` (+ `oob_r2` when available).
  - Optional CV MAE metrics (GroupKFold or KFold).
- `predict(features_dict)`
  - Dict → ordered vector → vector → target prediction.
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

Implements three subcommands:

### `collect` (6.2)

Builds graph list, extracts features, applies sparsification sweeps, times algorithm(s), writes CSV.

Input graph modes:

- `--graph-dir` for static `.txt` edge-list graphs.
- `--sample-file` for stream snapshot sampling.

Snapshot sampling (`sample_window_snapshots`) behavior:

- Reads all edges from stream file.
- Computes geometric progression of window sizes (`numpy.geomspace`) between a minimum and `max_edges` cap.
- Builds centered sliding windows and creates one `MockWindowManager` per window.

Collection behavior details:

- Generates multiple log-uniform `s_value` samples per graph.
- Applies `modifiedSpectralSparsity(s)` before timing.
- Times algorithm execution with `time.perf_counter()` and stores median (`numpy.median`) across repeats.
- Stores both pre/post features and derived `budget`.
- Stops further `s` sweeps early for a graph when sparsification no longer changes topology.

### `train` (6.3)

- Loads CSV rows.
- Filters rows to `--algo`.
- Builds `X/y`.
- Uses `s_value` as the current training target in `cmd_train`.
- Supports optional KFold cross-validation via `--cv-folds` (set `0` to disable).
- Fits predictor and prints MAE/R²/OOB (+ optional CV metrics).
- Prints feature importances.
- Saves artifacts into `--model-dir`.

### `run-all` (6.4)

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

See 4.4 for more details. Assuming working directory is `src/modelling_s`. 

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
python main.py train --csv timings.csv --algo pagerank --model-dir models --features pre_num_nodes,pre_num_edges,pre_log_num_nodes,pre_log_num_edges,pre_is_directed,budget
```

### 6.4 End-to-end in one command

```bash
python main.py run-all --sample-file ../../data/higgs-activity_time_postprocess.txt --algo pagerank --model-dir models --features pre_num_nodes,pre_num_edges,pre_log_num_nodes,pre_log_num_edges,pre_is_directed,budget
```

Note: inference is intentionally external to this CLI. The exported artifacts in `--model-dir` (`model.joblib`, `meta.json`) should be loaded by the runtime component that owns prediction.

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

Backward compatibility warning: existing `model.joblib` may be invalid if feature order or feature set changes.

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

- Data collection is expensive (sparsification sweeps + algorithm runs).
- Training is moderate cost and parallelized across CPU cores by scikit-learn.
- Prediction is low-latency (feature extraction + forest traversal).

## 8.3 Failure modes

- Empty or malformed graph file → graph load errors / no edges.
- Unsupported `--algo` value → registry `KeyError`.
- No matching algorithm rows during `train` → process exits.
- Missing artifacts in `model-dir` during `predict` → file load failure.
- Feature schema mismatch between training and prediction inputs → prediction-time validation or missing-feature errors.

---

## 9) Quick Reference

- Main CLI: `python main.py <collect|train|run-all> ...`
- Algorithms available: `list_algorithms()` in `algorithms.py`
- Feature order contract: `FEATURE_NAMES` in `feature_extraction.py`
- Model persistence entrypoint: `RuntimePredictor.save/load`

This module is designed to remain simple, inspectable, and easy to extend while enabling runtime- and sparsification-aware decision-making in the larger load-shedding system.
