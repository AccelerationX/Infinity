"""
Integration Test: Projects 7/8/11/12 with Real Tushare Data
"""

import sys
from pathlib import Path
import importlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "tushare_data"


def load_project(project_dir: Path, alias: str):
    """Load a project's core modules under a unique alias by temporarily manipulating sys.path."""
    sys.path.insert(0, str(project_dir))
    # Remove any previously imported 'core' modules to avoid cross-project contamination
    to_remove = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
    for k in to_remove:
        del sys.modules[k]

    models = importlib.import_module("core.models")
    engine = importlib.import_module("core.engine")

    sys.path.pop(0)
    sys.modules[f"{alias}_models"] = models
    sys.modules[f"{alias}_engine"] = engine
    return models, engine


# Load all four projects under unique aliases
p7_models, p7_engine = load_project(ROOT / "07_trade_attribution_analyzer", "p7")
p8_models, p8_engine = load_project(ROOT / "08_portfolio_risk_diagnosis", "p8")
p11_models, p11_engine = load_project(ROOT / "11_strategy_adversarial_robustness", "p11")

# Also load P8 risk_alerts separately
sys.path.insert(0, str(ROOT / "08_portfolio_risk_diagnosis"))
to_remove = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
for k in to_remove:
    del sys.modules[k]
p8_alerts = importlib.import_module("core.risk_alerts")
sys.path.pop(0)

# P12 needs its own engine load too
sys.path.insert(0, str(ROOT / "12_quant_research_workbench"))
to_remove = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
for k in to_remove:
    del sys.modules[k]
p12_engine_mod = importlib.import_module("core.engine")
sys.path.pop(0)

# ------------------------------------------------------------------
# 1. Load and preprocess Tushare data
# ------------------------------------------------------------------
def load_prices():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}.")

    price_dict = {}
    for f in files:
        df = pd.read_csv(f)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date").set_index("trade_date")
        symbol = f.stem.replace("_", ".")
        price_dict[symbol] = df["close"]

    prices = pd.DataFrame(price_dict)
    prices = prices.dropna()
    return prices


prices = load_prices()
print(f"[Data] Loaded prices for {prices.shape[1]} stocks, {prices.shape[0]} days")
print(f"       Date range: {prices.index[0].date()} ~ {prices.index[-1].date()}")

returns = prices.pct_change().dropna()

# ------------------------------------------------------------------
# 2. Simple momentum strategy
# ------------------------------------------------------------------
def run_momentum_strategy(prices_df: pd.DataFrame, lookback: int = 20, top_n: int = 2):
    rets = prices_df.pct_change().dropna()
    dates = rets.index
    rebalance_dates = dates[::lookback]

    weights = pd.DataFrame(0.0, index=dates, columns=prices_df.columns)
    strategy_rets = pd.Series(0.0, index=dates)

    for i, rd in enumerate(rebalance_dates):
        if i == 0:
            continue
        prev_rd = rebalance_dates[i - 1]
        mom = rets.loc[prev_rd:rd].sum()
        picks = mom.nlargest(top_n).index.tolist()

        mask = (dates >= rd) & (dates < rebalance_dates[i + 1]) if i + 1 < len(rebalance_dates) else (dates >= rd)
        period_dates = dates[mask]
        weights.loc[period_dates, picks] = 1.0 / top_n
        period_rets = rets.loc[period_dates]
        strategy_rets.loc[period_dates] = (period_rets * weights.loc[period_dates]).sum(axis=1)

    weights = weights.loc[strategy_rets != 0]
    strategy_rets = strategy_rets[strategy_rets != 0]
    nav = (1 + strategy_rets).cumprod()
    return weights, strategy_rets, nav


weights, strategy_rets, nav = run_momentum_strategy(prices, lookback=20, top_n=2)
print(f"[Strategy] Momentum strategy CAGR: {nav.iloc[-1] ** (252 / len(nav)) - 1:.2%}")
print(f"[Strategy] Max Drawdown: {(nav / nav.cummax() - 1).min():.2%}")
print(f"[Strategy] Sharpe: {strategy_rets.mean() / strategy_rets.std(ddof=1) * np.sqrt(252):.3f}")

# ------------------------------------------------------------------
# 3. Project 8: Portfolio Risk Diagnosis
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Project 8: Portfolio Risk Diagnosis")
print("=" * 60)

last_date = weights.index[-1]
snapshot = p8_models.RiskSnapshot(timestamp=last_date, weights=weights.loc[last_date])

factor_model = p8_models.FactorModel(
    factor_names=["MKT", "TECH", "CONS"],
    exposures=pd.DataFrame(
        {
            "MKT": [1.0, 1.0, 1.0, 1.0, 1.0],
            "TECH": [0.0, 0.0, 0.0, 1.0, 0.0],
            "CONS": [0.0, 0.0, 1.0, 0.0, 1.0],
        },
        index=prices.columns,
    ),
    factor_cov=pd.DataFrame(
        [[0.0003, 0.00005, 0.00003],
         [0.00005, 0.0004, 0.00002],
         [0.00003, 0.00002, 0.00025]],
        index=["MKT", "TECH", "CONS"],
        columns=["MKT", "TECH", "CONS"],
    ),
    idio_var=pd.Series({c: 0.0002 for c in prices.columns}),
)

engine8 = p8_engine.RiskDiagnosisEngine(p8_engine.EngineConfig(risk_free_rate=0.025))
engine8.load_returns_history(returns)
engine8.load_factor_model(factor_model)
engine8.load_factor_groups({"MARKET": ["MKT"], "SECTOR": ["TECH", "CONS"]})
engine8.load_alert_rules([
    p8_alerts.AlertRule("volatility", info_threshold=0.25, warning_threshold=0.35, direction="above"),
    p8_alerts.AlertRule("max_drawdown", warning_threshold=-0.20, direction="below"),
])

report8 = engine8.run(snapshot)

dec = report8.decomposition
if dec:
    print(f"Total Volatility : {dec.total_volatility:.4%}")
    for g, arc in dec.group_contributions.items():
        print(f"  {g:12s} ARC : {arc:.4%}")
    print(f"  Idiosyncratic    : {dec.idio_risk:.4%}")
    print(f"  Validates        : {dec.validate()}")

met = report8.metrics
if met:
    print(f"VaR (95%)        : {met.var_parametric_95:.4%}")
    print(f"CVaR (95%)       : {met.cvar_parametric_95:.4%}")
    print(f"Max Drawdown     : {met.max_drawdown:.4%}")
    print(f"Sharpe           : {met.sharpe_ratio:.3f}")

print("\nStress Scenarios:")
for sr in report8.stress_results:
    print(f"  {sr.scenario_name:20s}: P&L = {sr.expected_pnl:+.4%}")

if report8.alerts:
    print("\nAlerts:")
    for a in report8.alerts:
        print(f"  [{a.level.value}] {a.metric_name} = {a.metric_value:.4f}")
else:
    print("\nNo alerts triggered.")

# ------------------------------------------------------------------
# 4. Project 11: Strategy Adversarial Robustness
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Project 11: Strategy Adversarial Robustness")
print("=" * 60)

class MomentumStrategy(p11_models.Strategy):
    def __init__(self, prices_df):
        self.prices = prices_df

    def run(self, prices: pd.DataFrame, lookback: float = 20, top_n: float = 2, **kwargs) -> p11_models.StrategyResult:
        lookback = max(5, int(round(lookback)))
        top_n = max(1, int(round(top_n)))
        w, srets, nav = run_momentum_strategy(prices, lookback, top_n)
        if len(srets) == 0 or srets.std(ddof=1) == 0:
            return p11_models.StrategyResult(final_nav=1.0, total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0, returns=srets)
        sharpe = float(srets.mean() / srets.std(ddof=1) * np.sqrt(252))
        mdd = float((nav / nav.cummax() - 1).min())
        return p11_models.StrategyResult(
            final_nav=float(nav.iloc[-1]),
            total_return=float(nav.iloc[-1] - 1),
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            returns=srets,
            positions=w,
        )

strategy11 = MomentumStrategy(prices)
engine11 = p11_engine.RobustnessEngine(p11_engine.EngineConfig(seed=42))

report11 = engine11.run_full_analysis(
    strategy=strategy11,
    strategy_name="momentum_tushare",
    base_params={"lookback": 20, "top_n": 2},
    baseline_prices=prices,
    param_sensitivity_config={"lookback": [10, 20, 30, 40]},
    param_bounds={"lookback": (10.0, 50.0), "top_n": (1.0, 3.0)},
)

print(f"Robustness Rating : {report11.robustness_score.rating}")
print(f"Overall Score     : {report11.robustness_score.overall_score:.3f}")
print("Adversarial Results:")
for r in report11.adversarial_results:
    status = "SURVIVED" if r.survival else "FAILED"
    print(f"  {r.scenario_type:20s} (int={r.intensity:.1f}): NAV={r.strategy_result.final_nav:.3f} Sharpe={r.strategy_result.sharpe_ratio:.2f} [{status}]")

print("Failure Diagnoses:")
for d in report11.failure_diagnoses:
    print(f"  {d.mode.value:20s}: severity={d.severity_score:.2f}")

# ------------------------------------------------------------------
# 5. Project 7: Trade Attribution
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Project 7: Trade Attribution")
print("=" * 60)

rebalance_dates = weights.index[weights.diff().abs().sum(axis=1) > 0]
sample_dates = rebalance_dates[-4:-1]

signals = []
orders = []
fills = []
positions = []

for rd in sample_dates:
    day_weights = weights.loc[rd]
    held = day_weights[day_weights > 0].index.tolist()
    for sym in held:
        target_price = float(prices.loc[rd, sym])
        signals.append(p7_models.Signal(signal_time=rd, symbol=sym, target_weight=day_weights[sym], signal_price=target_price))
        orders.append(p7_models.Order(order_time=rd, symbol=sym, side=p7_models.Side.BUY, order_qty=100))
        fill_price = target_price * 1.0005
        fills.append(p7_models.Fill(fill_time=rd, symbol=sym, fill_qty=100, fill_price=fill_price, status=p7_models.FillStatus.FULLY_FILLED, fees=5.0))

last_rd = sample_dates[-1]
for sym in prices.columns:
    w = weights.loc[last_rd, sym]
    positions.append(p7_models.Position(date=last_rd, symbol=sym, shares=100 if w > 0 else 0, market_price=float(prices.loc[last_rd, sym])))

market_data = {}
for sym in prices.columns:
    md = prices[[sym]].rename(columns={sym: "close"}).copy()
    md["open"] = md["close"] * 0.995
    md["high"] = md["close"] * 1.01
    md["low"] = md["close"] * 0.99
    md["volume"] = 1e6
    md = md.reset_index().rename(columns={"trade_date": "date"})
    market_data[sym] = md

engine7 = p7_engine.AttributionEngine(p7_engine.EngineConfig(opportunity_horizon_days=5))
engine7.load_signals(signals)
engine7.load_orders(orders)
engine7.load_fills(fills)
engine7.load_positions(positions)
engine7.load_market_data(market_data)

benchmark_w = pd.Series({c: 0.2 for c in prices.columns})
next_rets = returns.loc[last_rd]

eval_prices = {}
for sym in prices.columns:
    try:
        eval_prices[sym] = float(prices.loc[prices.index[prices.index.get_loc(last_rd) + 1], sym])
    except IndexError:
        eval_prices[sym] = float(prices.loc[last_rd, sym])

report7 = engine7.run_full_analysis(
    period_label=str(last_rd.date()),
    benchmark_weights=benchmark_w,
    asset_returns=next_rets,
    evaluation_prices=eval_prices,
)

ret = report7["return_attribution"]
print(f"Period            : {ret.period}")
print(f"Total Return      : {ret.total_return:.4%}")
print(f"Selection Alpha   : {ret.selection_alpha:.4%}")
print(f"Execution Cost    : {ret.execution_cost:.4%}")
print(f"Validates         : {ret.validate()}")

exec_summary = report7["execution_summary"]
if not exec_summary.empty:
    print(f"Avg Slippage (bps): {exec_summary.loc['slippage_bps', 'mean']:.2f}")

isf = report7["implementation_shortfall"]
print(f"Implementation Shortfall: ${isf['shortfall_dollar']:.2f} ({isf['shortfall_bps']:.2f} bps)")

# ------------------------------------------------------------------
# 6. Project 12: Quant Research Workbench
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Project 12: Quant Research Workbench")
print("=" * 60)

engine12 = p12_engine_mod.WorkbenchEngine(root_dir=".workbench_tushare_integration")
exp_id = engine12.create_experiment(
    name="tushare_integration_test",
    description="End-to-end test of Projects 7/8/11/12 with real Tushare data",
    tags=["integration", "tushare", "momentum"],
)

with engine12.start_run(exp_id) as run:
    run.log_params({
        "data": {"stocks": list(prices.columns), "start": str(prices.index[0].date()), "end": str(prices.index[-1].date())},
        "strategy": {"type": "momentum", "lookback": 20, "top_n": 2},
    })
    run.log_metrics({
        "strategy_cagr": float(nav.iloc[-1] ** (252 / len(nav)) - 1),
        "strategy_sharpe": float(strategy_rets.mean() / strategy_rets.std(ddof=1) * np.sqrt(252)),
        "strategy_mdd": float((nav / nav.cummax() - 1).min()),
        "risk_total_volatility": float(report8.decomposition.total_volatility) if report8.decomposition else 0.0,
        "risk_var_95": float(report8.metrics.var_parametric_95) if report8.metrics else 0.0,
        "robustness_score": float(report11.robustness_score.overall_score),
        "execution_shortfall_bps": float(isf["shortfall_bps"]),
    })

comp = engine12.compare_experiment(exp_id)
print("Logged experiment comparison:")
print(comp[["run_id", "param.strategy.type", "metric.strategy_sharpe", "metric.robustness_score"]].to_string(index=False))

print("\n" + "=" * 60)
print("Integration test complete. All 4 projects ran successfully on real data.")
print("=" * 60)
