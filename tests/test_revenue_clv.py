"""BG/NBD and Gamma-Gamma: recovery of known parameters and correctness of the conditional
expectations, checked against data simulated from each model's own generative process."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from northstar.revenue.clv import (
    BGNBDParams,
    GammaGammaParams,
    bgnbd_expected_purchases,
    bgnbd_p_alive,
    fit_bgnbd,
    fit_gamma_gamma,
    gamma_gamma_expected_spend,
    purchase_summary,
)


def simulate_bgnbd(params: BGNBDParams, n: int, horizon: float, seed: int) -> pd.DataFrame:
    """Customers from the BG/NBD process: calibration summary (x, t_x, T) and future purchases.

    Each customer makes a first purchase at time 0, buys at Poisson rate ``lambda`` while alive
    and drops out right after any *repeat* purchase with probability ``p`` (so a customer with no
    repeat purchase is alive by construction, as in Fader, Hardie and Lee 2005).
    """
    rng = np.random.default_rng(seed)
    lam = rng.gamma(params.r, 1 / params.alpha, n)
    drop = rng.beta(params.a, params.b, n)
    T = rng.uniform(10, 70, n)
    rows = []
    for i in range(n):
        t, alive, times = 0.0, True, []
        while alive:
            t += rng.exponential(1 / lam[i])
            if t > T[i] + horizon:
                break
            times.append(t)
            alive = rng.random() >= drop[i]
        times = np.array(times)
        past = times[times <= T[i]]
        rows.append((len(past), past.max() if len(past) else 0.0, T[i],
                     int(((times > T[i]) & (times <= T[i] + horizon)).sum())))
    return pd.DataFrame(rows, columns=["x", "t_x", "T", "future"])


@pytest.fixture(scope="module", params=[BGNBDParams(r=0.6, alpha=4.0, a=1.6, b=4.0),
                                        BGNBDParams(r=0.5, alpha=3.5, a=0.3, b=2.0)],
                ids=["a>1", "a<1"])
def bgnbd_case(request):
    return request.param, simulate_bgnbd(request.param, n=6000, horizon=26.0, seed=5)


def test_bgnbd_recovers_known_parameters(bgnbd_case):
    true, sim = bgnbd_case
    fit = fit_bgnbd(sim["x"], sim["t_x"], sim["T"])
    assert fit.r == pytest.approx(true.r, rel=0.2)
    assert fit.alpha == pytest.approx(true.alpha, rel=0.25)
    # a and b are weakly identified separately; their implied mean dropout probability is not.
    assert fit.a / (fit.a + fit.b) == pytest.approx(true.a / (true.a + true.b), rel=0.25)


def test_bgnbd_conditional_expectation_matches_simulated_future(bgnbd_case):
    true, sim = bgnbd_case
    expected = bgnbd_expected_purchases(true, 26.0, sim["x"], sim["t_x"], sim["T"])
    assert np.all(np.isfinite(expected)) and np.all(expected >= 0)
    assert expected.mean() == pytest.approx(sim["future"].mean(), rel=0.06)
    # Conditioning is right, not only the average: check groups with different histories.
    groups = pd.cut(sim["x"], [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"])
    for label, g in sim.groupby(groups, observed=True):
        e = bgnbd_expected_purchases(true, 26.0, g["x"], g["t_x"], g["T"])
        assert e.mean() == pytest.approx(g["future"].mean(), rel=0.15, abs=0.05), label
    # A model fitted on the calibration data forecasts the total nearly as well.
    fit = fit_bgnbd(sim["x"], sim["t_x"], sim["T"])
    fitted = bgnbd_expected_purchases(fit, 26.0, sim["x"], sim["t_x"], sim["T"])
    assert fitted.sum() == pytest.approx(sim["future"].sum(), rel=0.1)


def test_p_alive_rules():
    params = BGNBDParams(r=0.5, alpha=4.0, a=0.8, b=2.5)
    assert bgnbd_p_alive(params, [0], [0.0], [30.0])[0] == pytest.approx(1.0)
    recent, stale = bgnbd_p_alive(params, [3, 3], [29.0, 5.0], [30.0, 30.0])
    assert 0 < stale < recent <= 1
    # For the same recency gap, a more frequent buyer who then goes quiet is likelier gone.
    few, many = bgnbd_p_alive(params, [1, 8], [10.0, 10.0], [30.0, 30.0])
    assert many < few
    more_time = bgnbd_expected_purchases(params, 52.0, [3], [29.0], [30.0])
    less_time = bgnbd_expected_purchases(params, 26.0, [3], [29.0], [30.0])
    assert more_time > less_time > 0


def test_fit_rejects_inconsistent_summaries():
    with pytest.raises(ValueError, match="inconsistent"):
        fit_bgnbd([1, 2], [5.0, 12.0], [10.0, 10.0])  # t_x > T
    with pytest.raises(ValueError, match="inconsistent"):
        fit_bgnbd([0], [3.0], [10.0])  # a last repeat purchase without any repeat purchase


def test_gamma_gamma_recovers_population_mean_and_shrinks_noisy_averages():
    rng = np.random.default_rng(3)
    true = GammaGammaParams(p=6.0, q=4.0, v=15.0)
    n = 4000
    nu = rng.gamma(true.q, 1 / true.v, n)
    x = 1 + rng.poisson(2.0, n)
    m = np.array([rng.gamma(true.p, 1 / nu[i], x[i]).mean() for i in range(n)])
    fit = fit_gamma_gamma(x, m)
    assert fit.population_mean == pytest.approx(true.population_mean, rel=0.1)
    assert fit.q > 1
    # The conditional expectation is closer to each customer's true mean spend than the raw
    # average (shrinkage), and equals the population mean for customers with no repeat purchase.
    true_mean = true.p / nu
    conditional = gamma_gamma_expected_spend(fit, x, m)
    assert np.mean(np.abs(conditional - true_mean)) < np.mean(np.abs(m - true_mean))
    assert gamma_gamma_expected_spend(fit, [0], [0.0])[0] == pytest.approx(fit.population_mean)


def test_gamma_gamma_needs_repeat_customers():
    with pytest.raises(ValueError, match="repeat customers"):
        fit_gamma_gamma([0, 0, 1], [0.0, 0.0, 20.0])


def test_purchase_summary_aggregates_days_and_excludes_first_purchase_from_monetary():
    c = pd.Timestamp("2025-07-01")
    day = pd.Timedelta(days=1)
    orders = pd.DataFrame({
        "customer_id": ["A", "A", "A", "A", "B", "A"],
        "order_ts": [c - 70 * day, c - 70 * day + pd.Timedelta(hours=3), c - 35 * day,
                     c - 14 * day, c - 7 * day, c],
        "net_amount": [10.0, 5.0, 30.0, 50.0, 20.0, 999.0],
    })
    s = purchase_summary(orders, c, ["A", "B"])
    a, b = s.loc["A"], s.loc["B"]
    # Two orders on day -70 are one purchase occasion; the order at the cutoff is ignored.
    assert a["x"] == 2
    assert a["t_x"] == pytest.approx(56 / 7)
    assert a["T"] == pytest.approx(70 / 7)
    assert a["monetary"] == pytest.approx((30 + 50) / 2)
    assert (b["x"], b["t_x"], b["monetary"]) == (0, 0, 0)
    assert b["T"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no order"):
        purchase_summary(orders, c, ["A", "Z"])
