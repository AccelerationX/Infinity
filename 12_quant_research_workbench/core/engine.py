"""Main workbench engine.

Orchestrates experiment tracking, comparison, and parameter search
in a unified interface.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.experiment_compare import build_comparison_table, bootstrap_significance, param_sensitivity_grid
from core.experiment_tracker import ExperimentTracker
from core.parameter_search import best_params, grid_search, random_search


class WorkbenchEngine:
    """Main engine for the quant research workbench."""

    def __init__(self, root_dir: str = ".workbench"):
        self.tracker = ExperimentTracker(root_dir)

    def create_experiment(self, name: str, description: str = "", tags: Optional[List[str]] = None) -> str:
        """Create a new experiment and return its ID."""
        exp = self.tracker.create_experiment(name, description, tags)
        return exp.experiment_id

    def start_run(self, experiment_id: str):
        """Start a tracked run (context manager)."""
        return self.tracker.start_run(experiment_id)

    def get_experiments(self) -> pd.DataFrame:
        """Return all experiments as a DataFrame."""
        exps = self.tracker.get_experiments()
        return pd.DataFrame([{"experiment_id": e.experiment_id, "name": e.name, "description": e.description, "created_at": e.created_at} for e in exps])

    def get_runs(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """Return runs as a DataFrame."""
        runs = self.tracker.get_runs(experiment_id)
        return pd.DataFrame([{"run_id": r.run_id, "experiment_id": r.experiment_id, "status": r.status.value, "duration_ms": r.duration_ms} for r in runs])

    def compare_experiment(self, experiment_id: str) -> pd.DataFrame:
        """Build a comparison table for all runs in an experiment."""
        runs = self.tracker.get_runs(experiment_id)
        run_ids = [r.run_id for r in runs]

        # Load params and metrics from JSONL
        params = self._load_jsonl_records("params.jsonl", run_ids)
        metrics = self._load_jsonl_records("metrics.jsonl", run_ids)
        runs_dicts = [{"run_id": r.run_id, "experiment_id": r.experiment_id, "status": r.status.value} for r in runs]

        return build_comparison_table(runs_dicts, params, metrics)

    def compare_significance(
        self,
        experiment_id: str,
        metric_name: str,
        param_name: str,
        value_a: Any,
        value_b: Any,
        n_bootstrap: int = 10_000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare two groups of runs differing in a single parameter using bootstrap."""
        df = self.compare_experiment(experiment_id)
        param_col = f"param.{param_name}"
        metric_col = f"metric.{metric_name}"

        if param_col not in df.columns or metric_col not in df.columns:
            raise ValueError(f"Missing column: {param_col} or {metric_col}")

        group_a = df[df[param_col] == value_a][metric_col].dropna().values
        group_b = df[df[param_col] == value_b][metric_col].dropna().values

        if len(group_a) < 2 or len(group_b) < 2:
            return {"error": "Insufficient data for significance test"}

        return bootstrap_significance(group_a, group_b, n_bootstrap=n_bootstrap, seed=seed)

    def run_grid_search(
        self,
        experiment_id: str,
        objective: Callable[[Dict[str, Any]], float],
        param_grid: Dict[str, List[Any]],
    ) -> pd.DataFrame:
        """Run grid search and log each trial as a run."""
        results = grid_search(objective, param_grid)
        for _, row in results.iterrows():
            params = {k: row[k] for k in param_grid}
            metric = row["metric"]
            with self.tracker.start_run(experiment_id) as run_ctx:
                run_ctx.log_params(params)
                run_ctx.log_metrics({"objective": metric})
        return results

    def run_random_search(
        self,
        experiment_id: str,
        objective: Callable[[Dict[str, Any]], float],
        bounds: Dict[str, tuple],
        n_iter: int = 50,
        seed: Optional[int] = None,
        discrete_params: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run random search and log each trial as a run."""
        results = random_search(objective, bounds, n_iter, seed, discrete_params)
        keys = list(bounds.keys())
        for _, row in results.iterrows():
            params = {k: row[k] for k in keys}
            metric = row["metric"]
            with self.tracker.start_run(experiment_id) as run_ctx:
                run_ctx.log_params(params)
                run_ctx.log_metrics({"objective": metric})
        return results

    def _load_jsonl_records(self, fname: str, run_ids: List[str]) -> List[Dict[str, Any]]:
        """Load JSONL records filtered by run_ids."""
        records = []
        run_id_set = set(run_ids)
        fpath = self.tracker.root / fname
        if not fpath.exists():
            return records
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = __import__("json").loads(line)
                if d.get("run_id") in run_id_set:
                    records.append(d)
        return records
