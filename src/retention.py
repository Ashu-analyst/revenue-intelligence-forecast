"""Post-sales analytics: GRR / NRR and a churn-risk model for Customer Success."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["usage_score", "support_tickets_90d", "nps", "tenure_months", "seats"]


def load_accounts(data_dir: str = "data", snapshot: str = "2026-09-30") -> pd.DataFrame:
    a = pd.read_csv(f"{data_dir}/accounts.csv", parse_dates=["start_date"])
    a["tenure_months"] = ((pd.Timestamp(snapshot) - a["start_date"]).dt.days / 30.4).round(1)
    return a


def retention_kpis(a: pd.DataFrame) -> dict[str, float]:
    """Simplified portfolio view: GRR = (start ARR − churned ARR) / start ARR;
    NRR additionally adds expansion ARR."""
    start = a["arr_eur"].sum()
    churned = a.loc[a["churned"], "arr_eur"].sum()
    expansion = a["expansion_eur"].sum()
    return {"start_arr": start, "GRR": (start - churned) / start, "NRR": (start - churned + expansion) / start,
            "logo_churn": a["churned"].mean()}


def churn_model(a: pd.DataFrame, seed: int = 42) -> dict:
    eligible = a[a["tenure_months"] > 3]  # brand-new customers can't churn yet
    x, y = eligible[FEATURES], eligible["churned"].astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    model.fit(x, y)
    coefs = {f: float(c) for f, c in zip(FEATURES, model[-1].coef_[0].round(3))}
    # Top-decile lift: how much more churn is in the 10% riskiest accounts than on average
    top = proba >= np.percentile(proba, 90)
    lift = y[top].mean() / y.mean()
    scored = eligible.assign(churn_risk=model.predict_proba(x)[:, 1])
    at_risk = scored[~scored["churned"]].nlargest(10, "churn_risk")[
        ["account_id", "segment", "arr_eur", "usage_score", "support_tickets_90d", "nps", "churn_risk"]]
    return {"auc_cv": float(auc), "top_decile_lift": float(lift), "coefficients": coefs,
            "n": int(len(eligible)), "at_risk": at_risk}
