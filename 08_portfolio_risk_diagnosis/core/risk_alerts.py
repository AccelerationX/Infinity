"""Risk alerting and threshold monitoring module.

Supports static thresholds and dynamic rolling-quantile thresholds
with three severity levels: INFO, WARNING, CRITICAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import AlertLevel, RiskAlert


@dataclass
class AlertRule:
    """A single alert rule configuration."""

    metric_name: str
    info_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    # Dynamic mode overrides static thresholds
    use_dynamic: bool = False
    history_series: Optional[pd.Series] = None
    info_quantile: float = 0.70
    warning_quantile: float = 0.85
    critical_quantile: float = 0.95
    direction: str = "above"  # 'above' means alert when metric >= threshold

    def get_thresholds(self) -> Dict[str, float]:
        """Resolve effective thresholds (static or dynamic)."""
        if not self.use_dynamic or self.history_series is None or self.history_series.empty:
            return {
                "INFO": self.info_threshold if self.info_threshold is not None else np.inf,
                "WARNING": self.warning_threshold if self.warning_threshold is not None else np.inf,
                "CRITICAL": self.critical_threshold if self.critical_threshold is not None else np.inf,
            }

        clean = self.history_series.dropna()
        return {
            "INFO": float(np.quantile(clean, self.info_quantile)),
            "WARNING": float(np.quantile(clean, self.warning_quantile)),
            "CRITICAL": float(np.quantile(clean, self.critical_quantile)),
        }


def evaluate_alert(
    timestamp,
    metric_value: float,
    rule: AlertRule,
) -> Optional[RiskAlert]:
    """Evaluate a single metric against its alert rule.

    Args:
        timestamp: Current timestamp.
        metric_value: Current metric value.
        rule: AlertRule configuration.

    Returns:
        RiskAlert if threshold is breached, else None.
    """
    thresholds = rule.get_thresholds()

    level = None
    triggered_value = None

    if rule.direction == "above":
        if metric_value >= thresholds["CRITICAL"]:
            level = AlertLevel.CRITICAL
            triggered_value = thresholds["CRITICAL"]
        elif metric_value >= thresholds["WARNING"]:
            level = AlertLevel.WARNING
            triggered_value = thresholds["WARNING"]
        elif metric_value >= thresholds["INFO"]:
            level = AlertLevel.INFO
            triggered_value = thresholds["INFO"]
    else:  # direction == "below"
        if metric_value <= thresholds["CRITICAL"]:
            level = AlertLevel.CRITICAL
            triggered_value = thresholds["CRITICAL"]
        elif metric_value <= thresholds["WARNING"]:
            level = AlertLevel.WARNING
            triggered_value = thresholds["WARNING"]
        elif metric_value <= thresholds["INFO"]:
            level = AlertLevel.INFO
            triggered_value = thresholds["INFO"]

    if level is None:
        return None

    msg = (
        f"{rule.metric_name} is {metric_value:.4f}, "
        f"breached {level.value} threshold ({triggered_value:.4f})"
    )

    return RiskAlert(
        timestamp=timestamp,
        metric_name=rule.metric_name,
        metric_value=metric_value,
        threshold_value=triggered_value,
        level=level,
        message=msg,
    )


def evaluate_all_alerts(
    timestamp,
    metrics: Dict[str, float],
    rules: List[AlertRule],
) -> List[RiskAlert]:
    """Evaluate multiple metrics against their respective rules."""
    alerts: List[RiskAlert] = []
    rule_map = {r.metric_name: r for r in rules}
    for name, value in metrics.items():
        rule = rule_map.get(name)
        if rule is None:
            continue
        alert = evaluate_alert(timestamp, value, rule)
        if alert is not None:
            alerts.append(alert)
    return alerts
