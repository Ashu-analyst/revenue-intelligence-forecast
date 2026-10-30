"""Build the board-ready markdown report + charts:  python -m src.report"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import forecast as fc  # noqa: E402
from .generate_data import generate  # noqa: E402
from .retention import churn_model, load_accounts, retention_kpis  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
BACKTEST_QUARTERS = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]
NEXT_QUARTER = pd.Timestamp("2026-10-01")
NAVY, BLUE, GREY, ORANGE = "#1b2a4a", "#4f8fd6", "#9aa5b1", "#e07b39"


def eur(x: float) -> str:
    return f"€{x / 1000:,.0f}k"


def funnel_table() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW leads AS SELECT * FROM read_csv_auto('{DATA / 'leads.csv'}')")
    con.execute(f"CREATE VIEW opportunities AS SELECT * FROM read_csv_auto('{DATA / 'opportunities.csv'}')")
    return con.execute((ROOT / "sql" / "funnel.sql").read_text()).df()


def main() -> None:
    if not (DATA / "leads.csv").exists():
        generate(DATA)
    REPORTS.mkdir(exist_ok=True)
    opps, hist = fc.load(str(DATA))

    # 1. Funnel (2025 cohorts, by channel)
    f = funnel_table()
    f25 = f[pd.to_datetime(f["cohort_quarter"]).dt.year == 2025].groupby("channel")[["leads", "mqls", "sqls", "opps", "won", "won_eur"]].sum()
    f25["lead→won %"] = (f25["won"] / f25["leads"] * 100).round(2)
    f25["win rate %"] = (f25["won"] / f25["opps"] * 100).round(1)
    f25["€ per lead"] = (f25["won_eur"] / f25["leads"]).round(0).astype(int)
    f25["won_eur"] = f25["won_eur"].astype(int)

    # 2. Backtest
    bt = fc.backtest(opps, hist, BACKTEST_QUARTERS)
    mape = {c: bt[f"ape_{c}"].mean() for c in ["ensemble", "calibrated_pipeline", "naive_history_pipeline", "crm_default_pipeline", "holt_winters"]}

    # 3. Next-quarter scenarios
    rates = fc.calibrated_win_rates(opps, hist, NEXT_QUARTER)
    sc = fc.scenarios(opps, hist, NEXT_QUARTER, rates)

    # 4. Retention
    acc = load_accounts(str(DATA))
    kpi = retention_kpis(acc)
    cm = churn_model(acc)

    # Charts
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bt["quarter"], bt["actual"] / 1e3, color=NAVY, lw=2.5, marker="o", label="Actual")
    ax.plot(bt["quarter"], bt["calibrated_pipeline"] / 1e3, color=BLUE, marker="o", label="Snapshot-calibrated pipeline + run-rate")
    ax.plot(bt["quarter"], bt["naive_history_pipeline"] / 1e3, color="#c9d6e8", ls="-.", marker="o", label="Naive historical stage rates")
    ax.plot(bt["quarter"], bt["crm_default_pipeline"] / 1e3, color=GREY, ls="--", marker="o", label="CRM default probabilities")
    ax.plot(bt["quarter"], bt["holt_winters"] / 1e3, color=ORANGE, ls=":", marker="o", label="Holt-Winters (time series)")
    ax.set_ylabel("Bookings (€k)"); ax.set_title("Quarterly bookings: forecast backtest")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "backtest.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    stages = list(rates)
    ax.bar([s for s in stages], [fc.CRM_DEFAULT[s] * 100 for s in stages], color=GREY, width=0.4, align="edge", label="CRM default")
    ax.bar([s for s in stages], [rates[s] * 100 for s in stages], color=BLUE, width=-0.4, align="edge", label="Snapshot-calibrated")
    ax.set_ylabel("P(win) %"); ax.set_title("Win probability by stage")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "stage_probabilities.png", dpi=150); plt.close(fig)

    # Markdown report
    btv = bt[["quarter", "actual", "ensemble", "calibrated_pipeline", "naive_history_pipeline", "crm_default_pipeline", "holt_winters"]].copy()
    for c in btv.columns[1:]:
        btv[c] = btv[c].map(eur)
    md = [
        "# Revenue Intelligence Report (synthetic data)",
        "",
        "## 1. Next-quarter bookings outlook (Q4 2026)",
        f"- Open pipeline closing in Q4: **{sc['open_deals']} deals**, weighted value **{eur(sc['weighted_pipeline'])}**",
        f"- Expected create-and-close run-rate: **{eur(sc['run_rate'])}**",
        f"- Scenarios (Monte Carlo, 5,000 runs): downside P10 **{eur(sc['downside_P10'])}** · base P50 **{eur(sc['base_P50'])}** · upside P90 **{eur(sc['upside_P90'])}**",
        "",
        "## 2. Forecast accuracy (backtest, 7 quarters)",
        "| Method | MAPE |", "|---|---|",
        f"| Ensemble (pipeline + Holt-Winters, averaged) | **{mape['ensemble']:.1%}** |",
        f"| Snapshot-calibrated pipeline + run-rate | {mape['calibrated_pipeline']:.1%} |",
        f"| Naive historical stage win rates + run-rate | {mape['naive_history_pipeline']:.1%} |",
        f"| CRM default stage probabilities + run-rate | {mape['crm_default_pipeline']:.1%} |",
        f"| Holt-Winters time series | {mape['holt_winters']:.1%} |",
        "",
        btv.to_markdown(index=False),
        "",
        "![backtest](backtest.png)",
        "",
        "Calibrated vs default stage probabilities:",
        "",
        "| Stage | CRM default | Calibrated |", "|---|---|---|",
        *[f"| {s} | {fc.CRM_DEFAULT[s]:.0%} | {rates[s]:.0%} |" for s in stages],
        "",
        "![stages](stage_probabilities.png)",
        "",
        "## 3. Funnel by channel (2025 lead cohorts)",
        f25.reset_index().to_markdown(index=False),
        "",
        "## 4. Retention",
        f"- GRR **{kpi['GRR']:.1%}** · NRR **{kpi['NRR']:.1%}** · logo churn {kpi['logo_churn']:.1%}",
        f"- Churn-risk model (logistic regression, 5-fold CV): AUC **{cm['auc_cv']:.2f}**, top-decile lift **{cm['top_decile_lift']:.1f}×** (n={cm['n']})",
        f"- Drivers (standardised coefficients, + = more churn): {', '.join(f'{k} {v:+.2f}' for k, v in cm['coefficients'].items())}",
        "",
        "Top 10 active accounts by churn risk (hand to Customer Success):",
        "",
        cm["at_risk"].assign(churn_risk=lambda d: (d.churn_risk * 100).round(0).astype(int).astype(str) + "%").to_markdown(index=False),
        "",
    ]
    (REPORTS / "report.md").write_text("\n".join(md), encoding="utf-8")
    bt.to_csv(REPORTS / "backtest.csv", index=False)
    print("\n".join(md[:20]))


if __name__ == "__main__":
    main()
