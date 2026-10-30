"""Generate a realistic, SYNTHETIC B2B SaaS CRM dataset (deterministic seed).

Tables (CRM-agnostic, mapped to Salesforce / HubSpot objects in the README):
  leads.csv          lead_id, created_date, channel, segment, mql_date, sql_date, opportunity_id
  opportunities.csv  opportunity_id, account_id, created_date, close_date, stage, is_won, amount_eur, segment, channel
  stage_history.csv  opportunity_id, stage, entered_date
  accounts.csv       account_id, segment, start_date, arr_eur, seats, usage_score, support_tickets_90d, nps, churned, expansion_eur
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2026-09-30")
STAGES = ["Discovery", "Qualification", "Proposal", "Negotiation", "Closed"]
CHANNELS = {"Inbound": 0.45, "Outbound": 0.25, "Partner": 0.15, "Events": 0.15}
SEGMENTS = {"Mid-Market": 0.6, "Enterprise": 0.25, "SMB": 0.15}
DEAL_SIZE = {"SMB": 9_000, "Mid-Market": 28_000, "Enterprise": 85_000}
MQL_RATE = {"Inbound": 0.38, "Outbound": 0.22, "Partner": 0.45, "Events": 0.30}
SQL_RATE = {"SMB": 0.45, "Mid-Market": 0.40, "Enterprise": 0.33}
OPP_RATE = 0.70
WIN_RATE = {"SMB": 0.32, "Mid-Market": 0.27, "Enterprise": 0.22}


def _seasonality(d: pd.Timestamp) -> float:
    # B2B pattern: strong Q4, weak August
    return {8: 0.75, 10: 1.1, 11: 1.2, 12: 1.15}.get(d.month, 1.0)


def generate(out_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(START, END, freq="D")
    leads, opps, history = [], [], []
    lead_n = opp_n = 0

    for d in days:
        months_in = (d.year - START.year) * 12 + d.month - 1
        growth = 1 + 0.018 * months_in  # ~22% yearly lead growth
        n = rng.poisson(14 * growth * _seasonality(d) * (0.35 if d.weekday() >= 5 else 1.0))
        for _ in range(n):
            lead_n += 1
            ch = rng.choice(list(CHANNELS), p=list(CHANNELS.values()))
            seg = rng.choice(list(SEGMENTS), p=list(SEGMENTS.values()))
            row = {"lead_id": f"L{lead_n:06d}", "created_date": d, "channel": ch, "segment": seg,
                   "mql_date": pd.NaT, "sql_date": pd.NaT, "opportunity_id": None}
            if rng.random() < MQL_RATE[ch]:
                row["mql_date"] = d + pd.Timedelta(days=int(rng.integers(1, 10)))
                if rng.random() < SQL_RATE[seg]:
                    row["sql_date"] = row["mql_date"] + pd.Timedelta(days=int(rng.integers(3, 20)))
                    if rng.random() < OPP_RATE:
                        opp_n += 1
                        oid = f"O{opp_n:05d}"
                        row["opportunity_id"] = oid
                        created = row["sql_date"] + pd.Timedelta(days=int(rng.integers(1, 7)))
                        if created > END:  # not yet converted at snapshot date
                            row["opportunity_id"] = None
                            opp_n -= 1
                            leads.append(row)
                            continue
                        cycle = int(rng.gamma(4, {"SMB": 9, "Mid-Market": 18, "Enterprise": 32}[seg]))
                        close = created + pd.Timedelta(days=max(cycle, 10))
                        # partners bring slightly better deals
                        p_win = WIN_RATE[seg] * (1.25 if ch == "Partner" else 1.0)
                        won = rng.random() < p_win
                        # how far did it get? lost deals drop out at some stage
                        reached = 4 if won else int(rng.choice([0, 1, 2, 3], p=[0.35, 0.3, 0.2, 0.15]))
                        amount = round(float(rng.lognormal(np.log(DEAL_SIZE[seg]), 0.45)), -2)
                        stage_dates = np.linspace(0, (close - created).days, 5).astype(int)
                        for s_idx in range(reached + 1):
                            entered = created + pd.Timedelta(days=int(stage_dates[s_idx]))
                            history.append({"opportunity_id": oid, "stage": STAGES[s_idx], "entered_date": entered})
                        # open deals: anything with close date after the snapshot date
                        if close > END:
                            elapsed = (END - created).days
                            idx = int(np.searchsorted(stage_dates, elapsed, side="right") - 1)
                            stage = STAGES[min(max(idx, 0), 3)]
                            history[:] = [h for h in history if not (h["opportunity_id"] == oid and h["entered_date"] > END)]
                            opps.append({"opportunity_id": oid, "account_id": None, "created_date": created,
                                         "close_date": close, "stage": stage, "is_won": None,
                                         "amount_eur": amount, "segment": seg, "channel": ch,
                                         "_true_won": won})
                        else:
                            opps.append({"opportunity_id": oid, "account_id": f"A{opp_n:05d}" if won else None,
                                         "created_date": created, "close_date": close,
                                         "stage": "Closed Won" if won else "Closed Lost", "is_won": won,
                                         "amount_eur": amount, "segment": seg, "channel": ch, "_true_won": won})
            leads.append(row)

    leads_df = pd.DataFrame(leads)
    opps_df = pd.DataFrame(opps)
    hist_df = pd.DataFrame(history)

    # Accounts = won deals; simulate usage and churn with a known relationship
    won = opps_df[opps_df["is_won"] == True].copy()  # noqa: E712
    n = len(won)
    usage = np.clip(rng.normal(62, 18, n), 0, 100)
    tickets = rng.poisson(3, n)
    nps = np.clip(rng.normal(30, 25, n), -100, 100)
    tenure_m = ((END - won["close_date"]).dt.days / 30.4).to_numpy()
    logit = -1.2 - 0.045 * (usage - 60) + 0.18 * (tickets - 3) - 0.015 * nps + 0.02 * np.minimum(tenure_m, 24) - 0.5
    churn_p = 1 / (1 + np.exp(-logit))
    churned = (rng.random(n) < churn_p) & (tenure_m > 3)
    expansion = np.where(~churned & (usage > 70), won["amount_eur"].to_numpy() * rng.uniform(0.05, 0.35, n), 0).round(-2)
    accounts = pd.DataFrame({
        "account_id": won["account_id"].to_numpy(), "segment": won["segment"].to_numpy(),
        "start_date": won["close_date"].to_numpy(), "arr_eur": won["amount_eur"].to_numpy(),
        "seats": np.maximum(1, (won["amount_eur"].to_numpy() / 900).astype(int)),
        "usage_score": usage.round(1), "support_tickets_90d": tickets, "nps": nps.round(0),
        "churned": churned, "expansion_eur": expansion,
    })

    truth = opps_df[["opportunity_id", "_true_won"]]
    opps_df = opps_df.drop(columns="_true_won")
    leads_df.to_csv(out / "leads.csv", index=False)
    opps_df.to_csv(out / "opportunities.csv", index=False)
    hist_df.to_csv(out / "stage_history.csv", index=False)
    accounts.to_csv(out / "accounts.csv", index=False)
    return {"leads": leads_df, "opportunities": opps_df, "stage_history": hist_df,
            "accounts": accounts, "_truth": truth}


if __name__ == "__main__":
    t = generate()
    for k, v in t.items():
        if not k.startswith("_"):
            print(f"{k:15s} {len(v):>7,} rows")
