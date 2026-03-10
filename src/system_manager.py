
import math
import time
from welford_variance import WelfordVariance


class TimeCard:
    """Lightweight container for recording the timing of one operation."""

    def __init__(self):
        self.start = None
        self.end = None
        self.count = None

    def __bool__(self):
        return (
            self.start is not None
            and self.end is not None
            and self.count is not None
        )

    @property
    def elapsed(self):
        """Wall-clock seconds spent (0.0 if not yet recorded)."""
        if self.start is not None and self.end is not None:
            return self.end - self.start
        return 0.0

    @property
    def cost_per_unit(self):
        """Per-item cost, or *None* when not meaningful."""
        if self and self.count > 0:
            return self.elapsed / self.count
        return None


class SystemManager:
    """Monitors per-window timing and decides how many edges to shed.

    Each processing cycle (one window slide) consists of:

        1. **Expire** old edges that fell out of the window.
        2. **Ingest** buffered edges into the graph.
        3. **Shed** edges if the predicted algorithm runtime would
           exceed the remaining time budget.
        4. **Run** the graph algorithm on the (possibly sparsified) graph.

    The total wall-clock time for steps 1-4 must fit within
    *slide_budget* seconds.
    """

    def __init__(self, slide_budget, headroom_frac=0.05, alpha=0.1,
                 runtime_predictor=None):
        """
        Parameters
        ----------
        slide_budget : float
            Wall-clock seconds available per processing cycle.
        headroom_frac : float
            Fraction of budget reserved as safety margin (0-1).
        alpha : float
            Cantelli confidence parameter.  Smaller values give more
            conservative (higher) upper-bound estimates of per-edge cost.
        runtime_predictor : object, optional
            An object with a ``.predict(features_dict) -> float`` method
            that estimates algorithm runtime from graph features.
        """
        self.log_path = "system_manager_log.txt"

        self.welfords_cpe_algo = WelfordVariance()
        self.welfords_cpe_expiry = WelfordVariance()
        self.welfords_cpe_ingest = WelfordVariance()

        self.slide_budget = slide_budget
        self.headroom_frac = headroom_frac
        self.alpha = alpha
        self.runtime_predictor = runtime_predictor

        self.expiry_card = TimeCard()
        self.ingest_card = TimeCard()
        self.algo_card = TimeCard()
        self.shed_card = TimeCard()

        with open(self.log_path, "w") as f:
            f.write(
                "window_count,"
                "expiry_time,expiry_cpe_mean,"
                "ingest_time,ingest_cpe_mean,"
                "algo_time,algo_cpe_mean,"
                "shed_count\n"
            )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def cantelli_upper_bound(self, wv: WelfordVariance) -> float:
        """Upper-bound on per-unit cost using Cantelli's inequality.

        P(X - mu >= k) <= var / (var + k^2)
        Setting the RHS equal to *alpha* and solving for *k*:
            k = sqrt(var * (1 - alpha) / alpha)
        """
        if wv.count < 2:
            return wv.mean if wv.count == 1 else 0.0
        k = math.sqrt(wv.get_variance() * (1 - self.alpha) / self.alpha)
        return wv.get_mean() + k

    def predict_algo_time(self, n_edges: int, graph_features: dict = None) -> float:
        """Predict algorithm runtime for the current graph.

        Uses *runtime_predictor* with *graph_features* when both are
        available; otherwise falls back to a linear estimate via
        Welford's per-edge cost.
        """
        if self.runtime_predictor is not None and graph_features is not None:
            try:
                return self.runtime_predictor.predict(graph_features)
            except Exception:
                pass  # fall through to Welford estimate

        if self.welfords_cpe_algo.count == 0:
            return 0.0  # no data yet, skip shedding on first window
        cpe = self.cantelli_upper_bound(self.welfords_cpe_algo)
        return cpe * n_edges

    # ------------------------------------------------------------------
    # Shedding decision
    # ------------------------------------------------------------------

    def compute_shed_count(self, n_edges: int,
                           expire_elapsed: float,
                           ingest_elapsed: float,
                           graph_features: dict = None) -> int:
        """Return the number of edges to shed so the algorithm finishes
        within the remaining time budget.

        Parameters
        ----------
        n_edges : int
            Number of unique edges currently in the window.
        expire_elapsed : float
            Wall-clock seconds already spent on expiry this cycle.
        ingest_elapsed : float
            Wall-clock seconds already spent on ingestion this cycle.
        graph_features : dict, optional
            Pre-extracted graph features for ML prediction.

        Returns
        -------
        int
            Number of edges to shed (0 means no shedding needed).
        """
        headroom = self.slide_budget * self.headroom_frac
        remaining = self.slide_budget - expire_elapsed - ingest_elapsed - headroom

        if remaining <= 0:
            return n_edges  # already over budget

        predicted_algo = self.predict_algo_time(n_edges, graph_features)
        if predicted_algo <= remaining:
            return 0  # algorithm fits, no shedding needed

        # Algorithm won't fit — figure out how many edges we can keep.
        if self.welfords_cpe_algo.count > 0:
            cpe = self.cantelli_upper_bound(self.welfords_cpe_algo)
            if cpe > 0:
                keep = int(remaining / cpe)
                return max(0, n_edges - keep)

        # Proportional fallback when no per-edge cost data is available.
        if predicted_algo > 0:
            ratio = remaining / predicted_algo
            keep = int(n_edges * ratio)
            return max(0, n_edges - keep)

        return 0

    # ------------------------------------------------------------------
    # End-of-cycle bookkeeping
    # ------------------------------------------------------------------

    def update_cycle_stats(self):
        """Update Welford running statistics from this cycle's TimeCards,
        write a log line, and reset cards for the next cycle."""
        if self.expiry_card and self.expiry_card.count > 0:
            self.welfords_cpe_expiry.add_var(self.expiry_card.cost_per_unit)
        if self.ingest_card and self.ingest_card.count > 0:
            self.welfords_cpe_ingest.add_var(self.ingest_card.cost_per_unit)
        if self.algo_card and self.algo_card.count > 0:
            self.welfords_cpe_algo.add_var(self.algo_card.cost_per_unit)

        def _safe_mean(wv):
            return wv.get_mean() if wv.count > 0 else 0.0

        with open(self.log_path, "a") as f:
            f.write(
                f"{self.welfords_cpe_algo.count},"
                f"{self.expiry_card.elapsed},"
                f"{_safe_mean(self.welfords_cpe_expiry)},"
                f"{self.ingest_card.elapsed},"
                f"{_safe_mean(self.welfords_cpe_ingest)},"
                f"{self.algo_card.elapsed},"
                f"{_safe_mean(self.welfords_cpe_algo)},"
                f"{self.shed_card.count or 0}\n"
            )

        # Reset for next cycle
        self.expiry_card = TimeCard()
        self.ingest_card = TimeCard()
        self.algo_card = TimeCard()
        self.shed_card = TimeCard()
