"""Tests for risk alerts module."""

import numpy as np
import pandas as pd
import pytest

from core.models import AlertLevel
from core.risk_alerts import AlertRule, evaluate_alert, evaluate_all_alerts


class TestRiskAlerts:
    def test_static_above_threshold_info(self):
        rule = AlertRule(
            metric_name="volatility",
            info_threshold=0.15,
            warning_threshold=0.20,
            critical_threshold=0.25,
            direction="above",
        )
        alert = evaluate_alert(pd.Timestamp("2024-01-01"), 0.16, rule)
        assert alert is not None
        assert alert.level == AlertLevel.INFO

    def test_static_above_threshold_warning(self):
        rule = AlertRule(
            metric_name="volatility",
            info_threshold=0.15,
            warning_threshold=0.20,
            critical_threshold=0.25,
            direction="above",
        )
        alert = evaluate_alert(pd.Timestamp("2024-01-01"), 0.21, rule)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_static_above_threshold_critical(self):
        rule = AlertRule(
            metric_name="volatility",
            info_threshold=0.15,
            warning_threshold=0.20,
            critical_threshold=0.25,
            direction="above",
        )
        alert = evaluate_alert(pd.Timestamp("2024-01-01"), 0.26, rule)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_no_alert(self):
        rule = AlertRule(
            metric_name="volatility",
            info_threshold=0.15,
            direction="above",
        )
        alert = evaluate_alert(pd.Timestamp("2024-01-01"), 0.10, rule)
        assert alert is None

    def test_below_direction(self):
        rule = AlertRule(
            metric_name="liquidity",
            info_threshold=0.50,
            warning_threshold=0.30,
            critical_threshold=0.10,
            direction="below",
        )
        alert = evaluate_alert(pd.Timestamp("2024-01-01"), 0.25, rule)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_dynamic_threshold(self):
        history = pd.Series(np.random.normal(0.10, 0.02, size=252))
        rule = AlertRule(
            metric_name="var",
            use_dynamic=True,
            history_series=history,
            info_quantile=0.70,
            warning_quantile=0.85,
            critical_quantile=0.95,
            direction="above",
        )
        q95 = np.quantile(history, 0.95)
        alert = evaluate_alert(pd.Timestamp("2024-01-01"), q95 + 0.001, rule)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_multiple_alerts(self):
        rules = [
            AlertRule("volatility", info_threshold=0.15, direction="above"),
            AlertRule("beta", info_threshold=1.0, direction="above"),
        ]
        metrics = {"volatility": 0.20, "beta": 0.80}
        alerts = evaluate_all_alerts(pd.Timestamp("2024-01-01"), metrics, rules)
        assert len(alerts) == 1
        assert alerts[0].metric_name == "volatility"
