"""End-to-end tests for the workbench engine."""

import tempfile

from core.engine import WorkbenchEngine


def test_engine_full_pipeline():
    with tempfile.TemporaryDirectory() as td:
        engine = WorkbenchEngine(root_dir=td)
        exp_id = engine.create_experiment("momentum_study", "Study momentum lookback", ["momentum"])

        def objective(params):
            lookback = params.get("lookback", 10)
            return float(lookback) * 0.01

        # Grid search
        results = engine.run_grid_search(
            experiment_id=exp_id,
            objective=objective,
            param_grid={"lookback": [5, 10, 20]},
        )
        assert len(results) == 3

        # Compare
        comp = engine.compare_experiment(exp_id)
        assert len(comp) == 3
        assert "param.lookback" in comp.columns
        assert "metric.objective" in comp.columns

        # Best params
        from core.parameter_search import best_params
        best = best_params(results)
        assert best["lookback"] == 20
