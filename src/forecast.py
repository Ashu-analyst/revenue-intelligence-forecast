"""Bookings forecasting: calibrated pipeline model + run-rate, with backtest and scenarios.

Method (per forecast quarter starting at snapshot date S):
  forecast = Σ_open deals closing in quarter  amount × P(win | stage at S)      (pipeline)
           + average in-quarter "create-and-close" bookings of the last 4 quarters (run-rate)

Win rates are learned only from outcomes known BEFORE S (no data leakage).
We compare four methods on the same quarters:
  1. snapshot-calibrated pipeline + run-rate   (P(win | stage the deal is in at quarter start))
  2. naive historical pipeline + run-rate      (P(win | deal ever reached the stage)) -> biased
  3. CRM default stage probabilities (10/25/50/75 %) + run-rate
  4. Holt-Winters time-series on monthly bookings (no pipeline information)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

STAGES = ["Discovery", "Qualification", "Proposal", "Negotiation"]
CRM_DEFAULT = {"Discovery": 0.10, "Qualification": 0.25, "Proposal": 0.50, "Negotiation": 0.75}


def load(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    opps = pd.read_csv(f"{data_dir}/opportunities.csv", parse_dates=["created_date", "close_date"])
    hist = pd.read_csv(f"{data_dir}/stage_history.csv", parse_dates=["entered_date"])
    return opps, hist


def stage_at(hist: pd.DataFrame, snapshot: pd.Timestamp) -> pd.Series:
    """Latest open stage each opportunity had reached on the snapshot date."""
    h = hist[(hist["entered_date"] <= snapshot) & (hist["stage"].isin(STAGES))]
    order = {s: i for i, s in enumerate(STAGES)}
    h = h.assign(o=h["stage"].map(order)).sort_values("o")
    return h.groupby("opportunity_id")["stage"].last()


def naive_win_rates(opps: pd.DataFrame, hist: pd.DataFrame, snapshot: pd.Timestamp) -> dict[str, float]:
    """P(win | deal EVER reached stage), from deals closed before the snapshot.
    Looks reasonable but is biased: a deal that is *still* in Discovery shortly before its
    close date is usually a stalled deal, not an average one. Kept for comparison."""
    closed = opps[(opps["close_date"] < snapshot) & opps["stage"].isin(["Closed Won", "Closed Lost"])]
    reached = hist[hist["opportunity_id"].isin(closed["opportunity_id"])]
    won_ids = set(closed.loc[closed["stage"] == "Closed Won", "opportunity_id"])
    rates = {}
    for s in STAGES:
        ids = reached.loc[reached["stage"] == s, "opportunity_id"].unique()
        rates[s] = float(np.mean([i in won_ids for i in ids])) if len(ids) else CRM_DEFAULT[s]
    return rates


def calibrated_win_rates(opps: pd.DataFrame, hist: pd.DataFrame, snapshot: pd.Timestamp,
                         lookback_quarters: int = 6) -> dict[str, float]:
    """Snapshot-based calibration: replay past quarter starts, take the deals that were open
    and due to close in that quarter, and measure how often each *stage-at-snapshot* won.
    Only quarters whose outcomes are fully known before `snapshot` are used (no leakage)."""
    wins: dict[str, list[bool]] = {s: [] for s in STAGES}
    for k in range(1, lookback_quarters + 1):
        past = snapshot - pd.offsets.QuarterBegin(k, startingMonth=1)
        if _quarter_bounds(past)[1] >= snapshot:
            continue
        p = open_pipeline(opps, hist, past)
        for stage, won in zip(p["stage_at_snapshot"], p["stage"] == "Closed Won"):
            wins[stage].append(bool(won))
    return {s: float(np.mean(v)) if len(v) >= 5 else CRM_DEFAULT[s] for s, v in wins.items()}


def _quarter_bounds(start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return start, start + pd.offsets.QuarterEnd(0)


def run_rate(opps: pd.DataFrame, snapshot: pd.Timestamp, lookback: int = 4) -> float:
    """Average bookings from deals created AND won inside the same quarter, last N quarters."""
    vals = []
    for k in range(1, lookback + 1):
        qs = snapshot - pd.offsets.QuarterBegin(k, startingMonth=1)
        qs, qe = _quarter_bounds(qs)
        m = (opps["stage"] == "Closed Won") & opps["created_date"].between(qs, qe) & opps["close_date"].between(qs, qe)
        vals.append(opps.loc[m, "amount_eur"].sum())
    return float(np.mean(vals))


def open_pipeline(opps: pd.DataFrame, hist: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    qs, qe = _quarter_bounds(snapshot)
    stages = stage_at(hist, snapshot)
    p = opps[(opps["created_date"] < snapshot) & (opps["close_date"] >= snapshot) & (opps["close_date"] <= qe)].copy()
    p["stage_at_snapshot"] = p["opportunity_id"].map(stages)
    return p.dropna(subset=["stage_at_snapshot"])


def pipeline_forecast(opps, hist, snapshot, rates: dict[str, float]) -> float:
    p = open_pipeline(opps, hist, snapshot)
    return float((p["amount_eur"] * p["stage_at_snapshot"].map(rates)).sum()) + run_rate(opps, snapshot)


def monthly_bookings(opps: pd.DataFrame) -> pd.Series:
    won = opps[opps["stage"] == "Closed Won"]
    return won.set_index("close_date")["amount_eur"].resample("MS").sum()


def holt_winters_forecast(opps: pd.DataFrame, snapshot: pd.Timestamp) -> float:
    y = monthly_bookings(opps)
    y = y[y.index < snapshot]
    # Seasonal model needs two full years of history; before that, fall back to trend only.
    seasonal = "add" if len(y) >= 24 else None
    model = ExponentialSmoothing(y, trend="add", seasonal=seasonal,
                                 seasonal_periods=12 if seasonal else None).fit()
    return float(model.forecast(3).clip(lower=0).sum())


def actual_bookings(opps: pd.DataFrame, quarter_start: pd.Timestamp) -> float:
    qs, qe = _quarter_bounds(quarter_start)
    m = (opps["stage"] == "Closed Won") & opps["close_date"].between(qs, qe)
    return float(opps.loc[m, "amount_eur"].sum())


def backtest(opps, hist, quarters: list[str]) -> pd.DataFrame:
    rows = []
    for q in quarters:
        s = pd.Timestamp(q)
        rows.append({
            "quarter": f"{s.year}-Q{s.quarter}",
            "actual": actual_bookings(opps, s),
            "calibrated_pipeline": pipeline_forecast(opps, hist, s, calibrated_win_rates(opps, hist, s)),
            "naive_history_pipeline": pipeline_forecast(opps, hist, s, naive_win_rates(opps, hist, s)),
            "crm_default_pipeline": pipeline_forecast(opps, hist, s, CRM_DEFAULT),
            "holt_winters": holt_winters_forecast(opps, s),
        })
    df = pd.DataFrame(rows)
    # Simple ensemble: pipeline view (bottom-up) and time-series view (top-down) averaged.
    df["ensemble"] = (df["calibrated_pipeline"] + df["holt_winters"]) / 2
    for c in ["ensemble", "calibrated_pipeline", "naive_history_pipeline", "crm_default_pipeline", "holt_winters"]:
        df[f"ape_{c}"] = (df[c] - df["actual"]).abs() / df["actual"]
    return df


def scenarios(opps, hist, snapshot, rates, n_sim: int = 5000, seed: int = 1) -> dict[str, float]:
    """Monte Carlo over deal outcomes → P10 / P50 / P90 for board scenario planning."""
    rng = np.random.default_rng(seed)
    p = open_pipeline(opps, hist, snapshot)
    probs = p["stage_at_snapshot"].map(rates).to_numpy()
    amounts = p["amount_eur"].to_numpy()
    rr = run_rate(opps, snapshot)
    # run-rate uncertainty: ±20% (normal), deal outcomes: Bernoulli
    sims = (rng.random((n_sim, len(p))) < probs) @ amounts + rng.normal(rr, 0.2 * rr, n_sim)
    return {"downside_P10": float(np.percentile(sims, 10)), "base_P50": float(np.percentile(sims, 50)),
            "upside_P90": float(np.percentile(sims, 90)), "open_deals": int(len(p)),
            "weighted_pipeline": float((amounts * probs).sum()), "run_rate": rr}
