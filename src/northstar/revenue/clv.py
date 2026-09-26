"""Probabilistic customer lifetime value: BG/NBD purchase counts x Gamma-Gamma spend per purchase.

The classic "buy 'til you die" model for non-contractual retail (Fader, Hardie and Lee 2005;
Fader and Hardie 2013), implemented directly from the published likelihoods with ``scipy``:

* **BG/NBD.** While alive, a customer buys at a Poisson rate ``lambda ~ Gamma(r, alpha)``; after
  each purchase they drop out with probability ``p ~ Beta(a, b)``. Inputs per customer are the
  number of *repeat* purchases ``x``, the time of the last one ``t_x`` and the age ``T``, all
  measured from the first purchase. Output: expected purchases in the next ``t`` periods and the
  probability the customer is still "alive".
* **Gamma-Gamma.** Average spend per repeat purchase varies across customers around a
  Gamma-distributed mean; the conditional expectation shrinks each customer's observed average
  towards the population mean, more strongly when ``x`` is small.

Both models are fit by maximum likelihood on history before a cutoff only; they need no labels,
so they can be refit at every scoring run without touching the outcome window. Purchases are
aggregated to calendar days (several orders on one day count as one purchase occasion) and time is
measured in weeks for numerical stability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

UNIT_DAYS = 7.0  # model time unit: weeks


@dataclass(frozen=True)
class BGNBDParams:
    r: float
    alpha: float
    a: float
    b: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GammaGammaParams:
    p: float
    q: float
    v: float

    @property
    def population_mean(self) -> float:
        """Mean spend per purchase across the population, ``p v / (q - 1)``."""
        return self.p * self.v / (self.q - 1)

    def as_dict(self) -> dict:
        return asdict(self)


def purchase_summary(orders: pd.DataFrame, cutoff: pd.Timestamp, ids: Iterable[str],
                     unit_days: float = UNIT_DAYS) -> pd.DataFrame:
    """Per-customer ``x``, ``t_x``, ``T`` (in ``unit_days``) and mean repeat spend at ``cutoff``.

    Only orders strictly before the cutoff are read. ``monetary`` is the mean spend of repeat
    purchase days (the first purchase is excluded, as in the Gamma-Gamma model) and 0 when
    ``x = 0``. Every id must have at least one order before the cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    ids = pd.Index(ids, name="customer_id")
    o = orders.loc[(orders["order_ts"] < cutoff) & orders["customer_id"].isin(ids)]
    daily = (o.assign(day=o["order_ts"].dt.floor("D"))
             .groupby(["customer_id", "day"], sort=True)["net_amount"].sum().reset_index())
    g = daily.groupby("customer_id")
    first, last, n = g["day"].min(), g["day"].max(), g.size()
    first_spend = g["net_amount"].first()
    total = g["net_amount"].sum()
    missing = ids.difference(n.index)
    if len(missing):
        raise ValueError(f"{len(missing)} customers have no order before {cutoff.date()}")
    x = (n - 1).astype(float)
    out = pd.DataFrame({
        "x": x,
        "t_x": (last - first).dt.days / unit_days,
        "T": (cutoff - first).dt.total_seconds() / 86_400 / unit_days,
        "monetary": ((total - first_spend) / x.where(x > 0)).fillna(0.0),
    })
    return out.reindex(ids)


# ---------------------------------------------------------------- BG/NBD
def bgnbd_log_likelihood(params: BGNBDParams, x: np.ndarray, t_x: np.ndarray, T: np.ndarray
                         ) -> np.ndarray:
    """Per-customer log-likelihood (Fader, Hardie and Lee 2005, eq. 6)."""
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + t_x),
                      -np.inf)
    return a1 + a2 + np.logaddexp(a3, a4)


def fit_bgnbd(x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> BGNBDParams:
    """Maximum-likelihood BG/NBD parameters (optimized on the log scale)."""
    x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
    if np.any(t_x > T + 1e-9) or np.any(x < 0) or np.any((x == 0) & (t_x > 0)):
        raise ValueError("inconsistent purchase summary: need 0 <= t_x <= T and t_x = 0 if x = 0")

    def nll(theta: np.ndarray) -> float:
        ll = bgnbd_log_likelihood(BGNBDParams(*np.exp(theta)), x, t_x, T)
        return -float(ll.mean())

    res = minimize(nll, x0=np.zeros(4), method="L-BFGS-B", bounds=[(-8.0, 8.0)] * 4)
    if not res.success:
        raise RuntimeError(f"BG/NBD fit did not converge: {res.message}")
    return BGNBDParams(*(float(v) for v in np.exp(res.x)))


def bgnbd_expected_purchases(params: BGNBDParams, t: float, x: np.ndarray, t_x: np.ndarray,
                             T: np.ndarray) -> np.ndarray:
    """E[purchases in (T, T + t] | x, t_x, T] (Fader, Hardie and Lee 2005, eq. 10)."""
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
    first = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
    return first / _dropout_odds(params, x, t_x, T)


def bgnbd_p_alive(params: BGNBDParams, x: np.ndarray, t_x: np.ndarray, T: np.ndarray
                  ) -> np.ndarray:
    """P(customer has not dropped out at T | x, t_x, T). Equals 1 when x = 0 by construction."""
    x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
    return 1.0 / _dropout_odds(params, x, t_x, T)


def _dropout_odds(params: BGNBDParams, x: np.ndarray, t_x: np.ndarray, T: np.ndarray
                  ) -> np.ndarray:
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    with np.errstate(divide="ignore", invalid="ignore"):
        term = (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return 1.0 + np.where(x > 0, term, 0.0)


# ---------------------------------------------------------------- Gamma-Gamma
def gamma_gamma_log_likelihood(params: GammaGammaParams, x: np.ndarray, m: np.ndarray
                               ) -> np.ndarray:
    """Per-customer log-likelihood of mean repeat spend ``m`` over ``x`` purchases (x >= 1)."""
    p, q, v = params.p, params.q, params.v
    x, m = np.asarray(x, float), np.asarray(m, float)
    px = p * x
    return (gammaln(px + q) - gammaln(px) - gammaln(q) + q * np.log(v) + (px - 1) * np.log(m)
            + px * np.log(x) - (px + q) * np.log(x * m + v))


def fit_gamma_gamma(x: np.ndarray, m: np.ndarray) -> GammaGammaParams:
    """Maximum-likelihood Gamma-Gamma parameters from repeat customers (``x >= 1``, ``m > 0``).

    ``q`` is parameterized as ``1 + exp(theta)`` so the population mean spend is always finite.
    """
    x, m = np.asarray(x, float), np.asarray(m, float)
    keep = (x >= 1) & (m > 0)
    if keep.sum() < 10:
        raise ValueError("need at least 10 repeat customers to fit the Gamma-Gamma model")
    x, m = x[keep], m[keep]

    def params_from(theta: np.ndarray) -> GammaGammaParams:
        return GammaGammaParams(p=float(np.exp(theta[0])), q=float(1 + np.exp(theta[1])),
                                v=float(np.exp(theta[2])))

    def nll(theta: np.ndarray) -> float:
        return -float(gamma_gamma_log_likelihood(params_from(theta), x, m).mean())

    x0 = np.array([0.0, 0.0, np.log(m.mean())])
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=[(-8.0, 8.0), (-8.0, 8.0), (-8.0, 16.0)])
    if not res.success:
        raise RuntimeError(f"Gamma-Gamma fit did not converge: {res.message}")
    return params_from(res.x)


def gamma_gamma_expected_spend(params: GammaGammaParams, x: np.ndarray, m: np.ndarray
                               ) -> np.ndarray:
    """E[spend per purchase | x, m]; the population mean when ``x = 0``."""
    x, m = np.asarray(x, float), np.asarray(m, float)
    return params.p * (params.v + x * m) / (params.p * x + params.q - 1)


# ---------------------------------------------------------------- combined
def clv_features(orders: pd.DataFrame, cutoff: pd.Timestamp, ids: Iterable[str],
                 horizon_days: int) -> tuple[pd.DataFrame, dict]:
    """Fit both models on history before ``cutoff`` and score ``ids`` for the next horizon.

    Returns per-customer ``bgnbd_p_alive``, ``bgnbd_expected_orders`` (purchase days in the
    horizon), ``gg_expected_order_value`` and their product ``clv_expected_revenue``, plus the
    fitted parameters.
    """
    s = purchase_summary(orders, cutoff, ids)
    bg = fit_bgnbd(s["x"], s["t_x"], s["T"])
    gg = fit_gamma_gamma(s["x"], s["monetary"])
    t = horizon_days / UNIT_DAYS
    expected_orders = bgnbd_expected_purchases(bg, t, s["x"], s["t_x"], s["T"])
    spend = gamma_gamma_expected_spend(gg, s["x"], s["monetary"])
    out = pd.DataFrame({
        "bgnbd_p_alive": bgnbd_p_alive(bg, s["x"], s["t_x"], s["T"]),
        "bgnbd_expected_orders": expected_orders,
        "gg_expected_order_value": spend,
        "clv_expected_revenue": expected_orders * spend,
    }, index=s.index)
    return out, {"bgnbd": bg.as_dict(), "gamma_gamma": gg.as_dict(),
                 "customers": len(s), "repeat_customers": int((s["x"] > 0).sum())}
