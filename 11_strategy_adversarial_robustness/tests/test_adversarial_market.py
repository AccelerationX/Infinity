"""Tests for adversarial market simulation module."""

import numpy as np
import pandas as pd

from core.adversarial_market import (
    generate_adversarial_library,
    generate_gbm_paths,
    make_correlation_crash,
    make_trend_reversal,
    make_volatility_jump,
)


class TestGBMPaths:
    def test_shape_and_columns(self):
        paths = generate_gbm_paths(
            n_steps=100,
            symbols=["A", "B"],
            mu=np.array([0.05, 0.05]),
            sigma=np.array([0.2, 0.2]),
            corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
            seed=42,
        )
        assert paths.shape == (100, 2)
        assert list(paths.columns) == ["A", "B"]
        assert np.allclose(paths.iloc[0].values, 100.0)

    def test_positive_prices(self):
        paths = generate_gbm_paths(
            n_steps=50,
            symbols=["A", "B", "C"],
            mu=np.array([0.0, 0.0, 0.0]),
            sigma=np.array([0.3, 0.3, 0.3]),
            corr=np.eye(3),
            seed=42,
        )
        assert np.all(paths.values > 0)


class TestTrendReversal:
    def test_reversal_direction(self):
        base = generate_gbm_paths(
            n_steps=100, symbols=["A"], mu=np.array([0.2]), sigma=np.array([0.05]),
            corr=np.array([[1.0]]), seed=42,
        )
        rev = make_trend_reversal(base, 50, intensity=1.0, seed=42)
        pre_return = float(np.log(rev.iloc[49] / rev.iloc[0]).iloc[0])
        post_return = float(np.log(rev.iloc[-1] / rev.iloc[49]).iloc[0])
        # Post-period should go opposite to pre-period trend on average
        assert post_return < -pre_return * 0.5


class TestVolatilityJump:
    def test_higher_post_volatility(self):
        base = generate_gbm_paths(
            n_steps=252, symbols=["A"], mu=np.array([0.0]), sigma=np.array([0.20 / np.sqrt(252)]),
            corr=np.array([[1.0]]), seed=42,
        )
        jumped = make_volatility_jump(base, 126, intensity=1.0, seed=42)
        pre_vol = float(np.log(jumped.iloc[:126]).diff().dropna().std().iloc[0])
        post_vol = float(np.log(jumped.iloc[126:]).diff().dropna().std().iloc[0])
        assert post_vol > pre_vol * 1.2


class TestCorrelationCrash:
    def test_higher_post_correlation(self):
        base = generate_gbm_paths(
            n_steps=100, symbols=["A", "B"], mu=np.array([0.0, 0.0]),
            sigma=np.array([0.1, 0.1]), corr=np.array([[1.0, 0.1], [0.1, 1.0]]),
            seed=42,
        )
        crashed = make_correlation_crash(base, 50, intensity=1.0, seed=42)
        post_rets = np.log(crashed.iloc[50:]).diff().dropna()
        post_corr = float(post_rets.corr().iloc[0, 1])
        assert post_corr > 0.7


class TestAdversarialLibrary:
    def test_library_size(self):
        lib = generate_adversarial_library(n_steps=100, symbols=["A", "B"], base_seed=42)
        # 3 intensities * 3 scenario types = 9 paths
        assert len(lib) == 9
        types = {p.scenario_type for p in lib}
        assert "TREND_REVERSAL" in types
        assert "VOLATILITY_JUMP" in types
        assert "CORRELATION_CRASH" in types
