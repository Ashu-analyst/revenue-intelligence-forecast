import pandas as pd
import pytest

from src import forecast as fc
from src.generate_data import generate
from src.retention import churn_model, load_accounts, retention_kpis


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d)
    opps, hist = fc.load(str(d))
    return d, opps, hist


def test_generator_is_deterministic(tmp_path, data):
    _, opps, _ = data
    generate(tmp_path)
    again = pd.read_csv(tmp_path / "opportunities.csv")
    assert len(again) == len(opps)


def test_no_leakage_in_calibration(data):
    """Win rates for a snapshot must not change if we delete everything after it."""
    _, opps, hist = data
    s = pd.Timestamp("2025-07-01")
    full = fc.calibrated_win_rates(opps, hist, s)
    cut_opps = opps[opps["created_date"] < s].copy()
    cut_opps.loc[cut_opps["close_date"] >= s, "stage"] = "open"  # outcome unknown at s
    cut_hist = hist[hist["entered_date"] < s]
    assert fc.calibrated_win_rates(cut_opps, cut_hist, s) == full


def test_win_rates_increase_with_stage(data):
    _, opps, hist = data
    r = fc.calibrated_win_rates(opps, hist, pd.Timestamp("2026-07-01"))
    assert r["Negotiation"] > r["Discovery"]
    assert all(0 <= v <= 1 for v in r.values())


def test_backtest_beats_naive_rates(data):
    _, opps, hist = data
    bt = fc.backtest(opps, hist, ["2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"])
    assert bt["ape_calibrated_pipeline"].mean() < bt["ape_naive_history_pipeline"].mean()


def test_scenarios_are_ordered(data):
    _, opps, hist = data
    s = pd.Timestamp("2026-10-01")
    sc = fc.scenarios(opps, hist, s, fc.calibrated_win_rates(opps, hist, s), n_sim=1000)
    assert sc["downside_P10"] < sc["base_P50"] < sc["upside_P90"]


def test_retention_and_churn_model(data):
    d, _, _ = data
    a = load_accounts(str(d))
    k = retention_kpis(a)
    assert 0 < k["GRR"] <= 1 and k["NRR"] >= k["GRR"]
    cm = churn_model(a)
    assert cm["auc_cv"] > 0.6
    assert cm["coefficients"]["usage_score"] < 0  # more usage -> less churn
