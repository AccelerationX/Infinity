"""
Batch Audit: Apply Projects 7/8/11/12 to Trading strategy lines.

Reads strategy_line_catalog.csv, locates historical detail.csv files,
and runs risk diagnosis, robustness testing, attribution (for main_live),
and logs everything into the research workbench.
"""

import sys
from pathlib import Path
import importlib
import numpy as np
import pandas as pd

TRADING_ROOT = Path(r"D:\Trading")
RESEARCH_ROOT = Path(__file__).parent

# ------------------------------------------------------------------
# Load research project modules dynamically
# ------------------------------------------------------------------
def load_project(project_dir: Path):
    sys.path.insert(0, str(project_dir))
    to_remove = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
    for k in to_remove:
        del sys.modules[k]
    models = importlib.import_module("core.models")
    engine = importlib.import_module("core.engine")
    sys.path.pop(0)
    return models, engine

# Load all 4 projects
p7_models, p7_engine = load_project(RESEARCH_ROOT / "07_trade_attribution_analyzer")
p8_models, p8_engine = load_project(RESEARCH_ROOT / "08_portfolio_risk_diagnosis")
p11_models, p11_engine = load_project(RESEARCH_ROOT / "11_strategy_adversarial_robustness")

# P8 risk_alerts
sys.path.insert(0, str(RESEARCH_ROOT / "08_portfolio_risk_diagnosis"))
to_remove = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
for k in to_remove:
    del sys.modules[k]
p8_alerts = importlib.import_module("core.risk_alerts")
sys.path.pop(0)

# P12
sys.path.insert(0, str(RESEARCH_ROOT / "12_quant_research_workbench"))
to_remove = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
for k in to_remove:
    del sys.modules[k]
p12_engine = importlib.import_module("core.engine")
sys.path.pop(0)

# ------------------------------------------------------------------
# Helper: locate detail.csv for a strategy line
# ------------------------------------------------------------------
def locate_detail_csv(project: str, strategy_label: str) -> Path | None:
    candidates = [
        TRADING_ROOT / "projects" / project / "outputs" / f"{strategy_label}_detail.csv",
        TRADING_ROOT / "projects" / project / "outputs" / f"{project}_detail.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: any *detail.csv in the project's outputs
    out_dir = TRADING_ROOT / "projects" / project / "outputs"
    if out_dir.exists():
        fallbacks = sorted(out_dir.glob("*detail.csv"))
        if fallbacks:
            return fallbacks[0]
    return None


def read_detail_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Project 8: Risk Diagnosis on return series
# ------------------------------------------------------------------
def run_risk_diagnosis(returns: pd.Series, label: str) -> dict:
    """Compute risk metrics from a strategy return series."""
    if len(returns) < 10:
        return {"error": "insufficient data"}

    # Trading detail.csv returns are period returns (not daily).
    # Detect the average period length to annualize correctly.
    avg_period_days = float(returns.index.to_series().diff().dt.days.dropna().mean())
    periods_per_year = 252.0 / max(avg_period_days, 1.0)

    mean_ret = float(returns.mean())
    vol = float(returns.std(ddof=1))
    rf_period = 0.025 / periods_per_year

    sharpe = (mean_ret - rf_period) / vol * (periods_per_year ** 0.5) if vol > 0 else 0.0
    downside = returns - rf_period
    downside_dev = float(np.sqrt(np.mean(np.minimum(downside, 0.0) ** 2)))
    sortino = (mean_ret - rf_period) / downside_dev if downside_dev > 0 else 0.0

    nav = (1 + returns).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())

    # VaR/CVaR on period returns (parametric)
    z95 = 1.6449
    var_95 = mean_ret - z95 * vol
    # Approximate CVaR parametric
    phi_z = 0.1031  # phi(1.6449)
    cvar_95 = mean_ret - vol * (phi_z / 0.05)

    return {
        "mean_return": mean_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_drawdown": mdd,
        "avg_period_days": avg_period_days,
        "periods_per_year": periods_per_year,
    }


# ------------------------------------------------------------------
# Project 11: Robustness / Failure Diagnosis
# ------------------------------------------------------------------
def run_failure_diagnosis(returns: pd.Series, label: str) -> dict:
    """Run failure mode diagnosis on a strategy return series."""
    nav = (1 + returns).cumprod()

    result = p11_models.StrategyResult(
        final_nav=float(nav.iloc[-1]),
        total_return=float(nav.iloc[-1] - 1),
        sharpe_ratio=float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std() > 0 else 0.0,
        max_drawdown=float((nav / nav.cummax() - 1).min()),
        returns=returns,
    )

    # Data snooping: split 70% IS / 30% OOS by time
    split_idx = int(len(returns) * 0.7)
    is_rets = returns.iloc[:split_idx]
    oos_rets = returns.iloc[split_idx:]

    is_nav = (1 + is_rets).cumprod()
    oos_nav = (1 + oos_rets).cumprod()

    insample = p11_models.StrategyResult(
        final_nav=float(is_nav.iloc[-1]) if len(is_nav) else 1.0,
        total_return=float(is_nav.iloc[-1] - 1) if len(is_nav) else 0.0,
        sharpe_ratio=float(is_rets.mean() / is_rets.std(ddof=1) * np.sqrt(252)) if len(is_rets) > 1 and is_rets.std() > 0 else 0.0,
        max_drawdown=float((is_nav / is_nav.cummax() - 1).min()) if len(is_nav) else 0.0,
        returns=is_rets,
    )

    outsample = p11_models.StrategyResult(
        final_nav=float(oos_nav.iloc[-1]) if len(oos_nav) else 1.0,
        total_return=float(oos_nav.iloc[-1] - 1) if len(oos_nav) else 0.0,
        sharpe_ratio=float(oos_rets.mean() / oos_rets.std(ddof=1) * np.sqrt(252)) if len(oos_rets) > 1 and oos_rets.std() > 0 else 0.0,
        max_drawdown=float((oos_nav / oos_nav.cummax() - 1).min()) if len(oos_nav) else 0.0,
        returns=oos_rets,
    )

    diagnoses = p11_engine.run_all_diagnoses(
        strategy_result=result,
        insample_result=insample,
        outsample_result=outsample,
    )

    return {
        d.mode.value: {
            "severity": d.severity_score,
            "description": d.description,
        }
        for d in diagnoses
    }


# ------------------------------------------------------------------
# Project 7: Trade Attribution (main_live only)
# ------------------------------------------------------------------
def run_trade_attribution(trade_log_path: str, line_id: str) -> dict:
    """Run trade attribution for a strategy line with trade logs."""
    log_path = Path(trade_log_path)
    if not log_path.exists() or log_path.stat().st_size < 500:
        return {"error": "insufficient trade log data"}

    log_df = pd.read_csv(log_path, on_bad_lines='skip')
    if log_df.empty or len(log_df) < 5:
        return {"error": "insufficient trade log rows"}

    # Map Trading log columns to our models
    # Columns: rebalance_date,is_rebalance_day,stock_code,stock_name,system_action,...
    log_df["rebalance_date"] = pd.to_datetime(log_df["rebalance_date"])

    signals = []
    orders = []
    fills = []
    positions = []

    # Use latest rebalance date for attribution snapshot
    latest_date = log_df["rebalance_date"].max()
    latest_day = log_df[log_df["rebalance_date"] == latest_date]

    for _, row in latest_day.iterrows():
        sym = row["stock_code"]
        action = str(row["system_action"]).lower()
        signal_price = float(row["system_signal"]) if pd.notna(row.get("system_signal")) else 10.0

        signals.append(p7_models.Signal(
            signal_time=latest_date,
            symbol=sym,
            target_weight=0.1,  # placeholder
            signal_price=signal_price,
        ))

        side = p7_models.Side.BUY if action in ["buy", "keep"] else p7_models.Side.SELL
        orders.append(p7_models.Order(
            order_time=latest_date,
            symbol=sym,
            side=side,
            order_qty=100,
        ))

        actual_action = str(row.get("actual_action", "")).lower()
        if actual_action != "cancelled":
            fill_price = float(row["actual_price"]) if pd.notna(row.get("actual_price")) else signal_price
            fills.append(p7_models.Fill(
                fill_time=latest_date,
                symbol=sym,
                fill_qty=100,
                fill_price=fill_price,
                status=p7_models.FillStatus.FULLY_FILLED,
                fees=5.0,
            ))

    # Dummy market data
    market_data = {}
    for sym in log_df["stock_code"].unique():
        market_data[sym] = pd.DataFrame({
            "date": [latest_date],
            "open": [10.0],
            "high": [10.5],
            "low": [9.5],
            "close": [10.0],
            "volume": [1e6],
        })

    engine7 = p7_engine.AttributionEngine(p7_engine.EngineConfig())
    engine7.load_signals(signals)
    engine7.load_orders(orders)
    engine7.load_fills(fills)
    engine7.load_market_data(market_data)

    # Build dummy positions
    for sym in log_df["stock_code"].unique():
        positions.append(p7_models.Position(
            date=latest_date,
            symbol=sym,
            shares=100,
            market_price=10.0,
        ))
    engine7.load_positions(positions)

    symbols = list(log_df["stock_code"].unique())
    benchmark_w = pd.Series({s: 1.0 / len(symbols) for s in symbols})
    asset_returns = pd.Series({s: 0.0 for s in symbols})  # no next-day return in log
    eval_prices = {s: 10.0 for s in symbols}

    try:
        report = engine7.run_full_analysis(
            period_label=str(latest_date.date()),
            benchmark_weights=benchmark_w,
            asset_returns=asset_returns,
            evaluation_prices=eval_prices,
        )
        ret = report["return_attribution"]
        return {
            "total_return": ret.total_return,
            "execution_cost": ret.execution_cost,
            "shortfall_bps": report["implementation_shortfall"]["shortfall_bps"],
            "validates": ret.validate(),
        }
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# Main batch audit loop
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Trading Strategy Line Batch Audit")
    print("=" * 70)

    catalog_path = TRADING_ROOT / "outputs" / "strategy_line_catalog.csv"
    catalog = pd.read_csv(catalog_path)

    # Setup workbench
    workbench = p12_engine.WorkbenchEngine(root_dir=".trading_strategy_audit")
    exp_id = workbench.create_experiment(
        name="trading_strategy_batch_audit",
        description="Batch risk/robustness/attribution audit for all Trading strategy lines",
        tags=["trading", "audit", "risk", "robustness"],
    )

    summary_rows = []

    for _, row in catalog.iterrows():
        line_id = row["line_id"]
        project = row["project"]
        strategy_label = row["strategy_label"]
        trade_log = row.get("trade_log", "")

        print(f"\n>> Auditing: {line_id} ({strategy_label})")

        detail_path = locate_detail_csv(project, strategy_label)
        if not detail_path:
            print(f"   SKIP: No detail.csv found")
            continue

        detail = read_detail_csv(detail_path)
        if "net_return" not in detail.columns:
            print(f"   SKIP: detail.csv missing net_return column")
            continue

        returns = detail.set_index("trade_date")["net_return"]

        # --- Project 8: Risk Diagnosis ---
        risk = run_risk_diagnosis(returns, line_id)
        print(f"   Risk -> Sharpe={risk.get('sharpe', 0):.3f}, MDD={risk.get('max_drawdown', 0):.2%}, VaR95={risk.get('var_95', 0):.2%}")

        # --- Project 11: Failure Diagnosis ---
        failures = run_failure_diagnosis(returns, line_id)
        for mode, info in failures.items():
            print(f"   Failure -> {mode}: severity={info['severity']:.2f}")

        # --- Project 7: Trade Attribution (main_live only) ---
        attribution = {"status": "skipped"}
        if line_id == "main_live" and trade_log:
            attribution = run_trade_attribution(trade_log, line_id)
            print(f"   Attribution -> validates={attribution.get('validates', False)}, shortfall={attribution.get('shortfall_bps', 0):.2f} bps")

        # --- Project 12: Log to workbench ---
        with workbench.start_run(exp_id) as run:
            run.log_params({
                "line_id": line_id,
                "project": project,
                "strategy_label": strategy_label,
                "data_source": str(detail_path.relative_to(TRADING_ROOT)),
                "n_observations": len(returns),
                "date_start": str(returns.index[0].date()),
                "date_end": str(returns.index[-1].date()),
            })

            metrics_to_log = {
                "risk_sharpe": risk.get("sharpe", 0),
                "risk_sortino": risk.get("sortino", 0),
                "risk_volatility": risk.get("volatility", 0),
                "risk_var_95": risk.get("var_95", 0),
                "risk_cvar_95": risk.get("cvar_95", 0),
                "risk_max_drawdown": risk.get("max_drawdown", 0),
            }
            for mode, info in failures.items():
                metrics_to_log[f"failure_{mode.lower()}_severity"] = info["severity"]

            if "shortfall_bps" in attribution:
                metrics_to_log["attribution_shortfall_bps"] = attribution["shortfall_bps"]
                metrics_to_log["attribution_validates"] = 1.0 if attribution.get("validates") else 0.0

            run.log_metrics(metrics_to_log)

        summary_rows.append({
            "line_id": line_id,
            "n_obs": len(returns),
            "sharpe": risk.get("sharpe"),
            "mdd": risk.get("max_drawdown"),
            "var_95": risk.get("var_95"),
            "regime_breakdown": failures.get("REGIME_BREAKDOWN", {}).get("severity"),
            "crowding_crash": failures.get("CROWDING_CRASH", {}).get("severity"),
            "data_snooping": failures.get("DATA_SNOOPING", {}).get("severity"),
        })

    # Print summary
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Workbench comparison
    print("\n" + "=" * 70)
    print("WORKBENCH COMPARISON TABLE")
    print("=" * 70)
    comp = workbench.compare_experiment(exp_id)
    display_cols = [c for c in comp.columns if c.startswith("param.") or c.startswith("metric.")]
    print(comp[["run_id"] + display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("Batch audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
