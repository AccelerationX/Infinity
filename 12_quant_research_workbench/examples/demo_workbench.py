"""
Demo: Quant Research Workbench.

Demonstrates experiment creation, parameter tracking, metric logging,
grid search, and experiment comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import WorkbenchEngine


def main():
    print("=" * 60)
    print("Quant Research Workbench - Demo")
    print("=" * 60)

    engine = WorkbenchEngine(root_dir=".workbench_demo")

    # Create experiment
    exp_id = engine.create_experiment(
        name="momentum_lookback_study",
        description="Study the effect of lookback window on momentum strategy performance.",
        tags=["momentum", "parameter_sweep"],
    )
    print(f"\nCreated experiment: {exp_id}")

    # Define a toy objective
    def objective(params):
        lookback = params.get("lookback", 10)
        # Toy relationship: lookback=15 is best
        return 1.0 - 0.02 * abs(lookback - 15)

    # Run grid search and auto-log each trial
    print("\nRunning grid search...")
    results = engine.run_grid_search(
        experiment_id=exp_id,
        objective=objective,
        param_grid={"lookback": [5, 10, 15, 20, 30]},
    )
    print(results.to_string(index=False))

    # Compare all runs in the experiment
    print("\n[Experiment Comparison Table]")
    comp = engine.compare_experiment(exp_id)
    print(comp[["run_id", "param.lookback", "metric.objective"]].to_string(index=False))

    # Run random search
    print("\nRunning random search...")
    rand_results = engine.run_random_search(
        experiment_id=exp_id,
        objective=objective,
        bounds={"lookback": (5.0, 30.0)},
        n_iter=5,
        seed=42,
        discrete_params=["lookback"],
    )
    print(rand_results.to_string(index=False))

    # List experiments
    print("\n[All Experiments]")
    exps = engine.get_experiments()
    print(exps[["experiment_id", "name"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Demo complete. Data stored in .workbench_demo/")
    print("=" * 60)


if __name__ == "__main__":
    main()
