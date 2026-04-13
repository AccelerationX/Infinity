"""Experiment tracking module.

Provides a lightweight, file-based experiment tracking system
with context-manager style run logging.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import Artifact, Experiment, MetricSnapshot, ParamSnapshot, Run, RunStatus


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict into dot-notation keys."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _compute_checksum(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class ExperimentTracker:
    """Lightweight file-based experiment tracker."""

    def __init__(self, root_dir: str = ".workbench"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        for fname in ["experiments.jsonl", "runs.jsonl", "params.jsonl", "metrics.jsonl", "artifacts.jsonl"]:
            fpath = self.root / fname
            if not fpath.exists():
                fpath.touch()

    def _append_jsonl(self, fname: str, record: Dict[str, Any]) -> None:
        with open(self.root / fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def create_experiment(self, name: str, description: str = "", tags: Optional[List[str]] = None) -> Experiment:
        exp = Experiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            description=description,
            tags=tags or [],
        )
        self._append_jsonl("experiments.jsonl", asdict(exp))
        return exp

    @contextmanager
    def start_run(self, experiment_id: str):
        run = Run(
            run_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            status=RunStatus.RUNNING,
            start_time=datetime.now(),
        )
        self._append_jsonl("runs.jsonl", {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": None,
            "duration_ms": None,
        })
        ctx = _RunContext(self, run.run_id)
        try:
            yield ctx
            run.status = RunStatus.COMPLETED
        except Exception:
            run.status = RunStatus.FAILED
            raise
        finally:
            run.end_time = datetime.now()
            run.duration_ms = int((run.end_time - run.start_time).total_seconds() * 1000) if run.start_time else None
            self._update_run_status(run)

    def _update_run_status(self, run: Run) -> None:
        # Simple append-only update: write a new run record that overwrites logic is done at read time
        self._append_jsonl("runs.jsonl", {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration_ms": run.duration_ms,
        })

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        flat = _flatten_dict(params)
        for k, v in flat.items():
            self._append_jsonl("params.jsonl", {
                "run_id": run_id,
                "param_name": k,
                "param_value": v,
                "param_type": type(v).__name__,
            })

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self._append_jsonl("metrics.jsonl", {
                "run_id": run_id,
                "metric_name": k,
                "metric_value": float(v),
                "step": step,
                "timestamp": datetime.now().isoformat(),
            })

    def log_artifact(self, run_id: str, artifact_path: str, artifact_type: str = "") -> None:
        path = Path(artifact_path)
        checksum = _compute_checksum(path) if path.exists() else ""
        size = path.stat().st_size if path.exists() else 0
        self._append_jsonl("artifacts.jsonl", {
            "run_id": run_id,
            "artifact_path": str(path.resolve()),
            "artifact_type": artifact_type,
            "checksum": checksum,
            "size_bytes": size,
        })

    def get_experiments(self) -> List[Experiment]:
        exps: Dict[str, Experiment] = {}
        fpath = self.root / "experiments.jsonl"
        if not fpath.exists():
            return []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                exp = Experiment(
                    experiment_id=d["experiment_id"],
                    name=d["name"],
                    description=d.get("description", ""),
                    created_at=datetime.fromisoformat(d["created_at"]),
                    tags=d.get("tags", []),
                )
                exps[exp.experiment_id] = exp
        return list(exps.values())

    def get_runs(self, experiment_id: Optional[str] = None) -> List[Run]:
        runs: Dict[str, Run] = {}
        fpath = self.root / "runs.jsonl"
        if not fpath.exists():
            return []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                run_id = d["run_id"]
                runs[run_id] = Run(
                    run_id=run_id,
                    experiment_id=d["experiment_id"],
                    status=RunStatus(d["status"]),
                    start_time=datetime.fromisoformat(d["start_time"]) if d.get("start_time") else None,
                    end_time=datetime.fromisoformat(d["end_time"]) if d.get("end_time") else None,
                    duration_ms=d.get("duration_ms"),
                )
        if experiment_id:
            return [r for r in runs.values() if r.experiment_id == experiment_id]
        return list(runs.values())


class _RunContext:
    """Context object yielded by start_run for logging within a run scope."""

    def __init__(self, tracker: ExperimentTracker, run_id: str):
        self.tracker = tracker
        self.run_id = run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        self.tracker.log_params(self.run_id, params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.tracker.log_metrics(self.run_id, metrics, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str = "") -> None:
        self.tracker.log_artifact(self.run_id, artifact_path, artifact_type)
