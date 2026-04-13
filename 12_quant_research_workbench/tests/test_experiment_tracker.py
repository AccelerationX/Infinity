"""Tests for experiment tracker module."""

import tempfile
from pathlib import Path

import pytest

from core.experiment_tracker import ExperimentTracker, _flatten_dict
from core.models import RunStatus


class TestFlattenDict:
    def test_simple(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        assert _flatten_dict({"model": {"lr": 0.01}}) == {"model.lr": 0.01}


class TestExperimentTracker:
    def test_create_experiment(self):
        with tempfile.TemporaryDirectory() as td:
            tracker = ExperimentTracker(root_dir=td)
            exp = tracker.create_experiment("test_exp", "desc", ["tag1"])
            assert exp.name == "test_exp"
            assert exp.description == "desc"
            assert "tag1" in exp.tags

    def test_start_run_completes(self):
        with tempfile.TemporaryDirectory() as td:
            tracker = ExperimentTracker(root_dir=td)
            exp = tracker.create_experiment("test_exp")
            with tracker.start_run(exp.experiment_id) as run:
                run.log_params({"a": 1})
                run.log_metrics({"m": 0.5})
            runs = tracker.get_runs(exp.experiment_id)
            assert len(runs) == 1
            assert runs[0].status == RunStatus.COMPLETED

    def test_run_failure_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            tracker = ExperimentTracker(root_dir=td)
            exp = tracker.create_experiment("test_exp")
            with pytest.raises(ValueError):
                with tracker.start_run(exp.experiment_id) as run:
                    run.log_params({"a": 1})
                    raise ValueError("boom")
            runs = tracker.get_runs(exp.experiment_id)
            assert len(runs) == 1
            assert runs[0].status == RunStatus.FAILED

    def test_log_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            tracker = ExperimentTracker(root_dir=td)
            exp = tracker.create_experiment("test_exp")
            # Create a dummy file
            dummy = Path(td) / "dummy.txt"
            dummy.write_text("hello")
            with tracker.start_run(exp.experiment_id) as run:
                run.log_artifact(str(dummy), "text")
            # Verify artifact record exists
            artifacts = list(Path(td).glob("*.jsonl"))
            assert any(a.name == "artifacts.jsonl" for a in artifacts)
