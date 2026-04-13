"""Core data models for the quant research workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Experiment:
    """A research experiment definition."""

    experiment_id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class Run:
    """A single execution run within an experiment."""

    run_id: str
    experiment_id: str
    status: RunStatus = RunStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None


@dataclass
class ParamSnapshot:
    """A parameter value recorded for a run."""

    run_id: str
    param_name: str
    param_value: Any
    param_type: str = ""


@dataclass
class MetricSnapshot:
    """A metric value recorded for a run."""

    run_id: str
    metric_name: str
    metric_value: float
    step: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Artifact:
    """An artifact (file) produced by a run."""

    run_id: str
    artifact_path: str
    artifact_type: str = ""
    checksum: str = ""
    size_bytes: int = 0
