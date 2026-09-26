"""Frequentist tools for a two-arm experiment, kept small and explicit so each can be unit tested.

Every test is paired with the confidence interval built from the *same* standard error and
reference distribution, so "p < alpha" and "the CI excludes 0" can never disagree.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import optimize
from scipy import stats as st


def z_critical(alpha: float) -> float:
    return float(st.norm.ppf(1 - alpha / 2))


def two_proportion_test(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> dict:
    """Difference in proportions (treatment - control): unpooled Wald z-test and matching CI.

    The relative lift uses the delta method on the log risk ratio. ``p_value_pooled`` is the
    classical pooled-variance z-test (equal to Pearson's chi-square without continuity
    correction), reported only as a cross-check.
    """
    if min(n_c, n_t) <= 0:
        raise ValueError("both arms need at least one unit")
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    se = float(np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t))
    z_crit = z_critical(alpha)
    z = diff / se if se > 0 else 0.0
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = float(np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t)))
    z_pooled = diff / se_pooled if se_pooled > 0 else 0.0
    out = {
        "control": p_c, "treatment": p_t, "n_control": n_c, "n_treatment": n_t,
        "diff": diff, "se": se, "ci": [diff - z_crit * se, diff + z_crit * se],
        "z": z, "p_value": float(2 * st.norm.sf(abs(z))),
        "p_value_pooled": float(2 * st.norm.sf(abs(z_pooled))),
        "relative": None, "relative_ci": None,
    }
    if x_c > 0 and x_t > 0:
        log_rr = np.log(p_t / p_c)
        se_log = np.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c)
        out["relative"] = float(np.expm1(log_rr))
        out["relative_ci"] = [float(np.expm1(log_rr - z_crit * se_log)),
                              float(np.expm1(log_rr + z_crit * se_log))]
    return out


def welch_test(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> dict:
    """Difference in means with Welch's unequal-variance t-test and the matching t interval.

    The relative change uses the delta method on the log ratio of means.
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if min(len(c), len(t)) < 2:
        raise ValueError("both arms need at least two observations")
    m_c, m_t = c.mean(), t.mean()
    v_c, v_t = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = float(np.sqrt(v_c + v_t))
    df = float((v_c + v_t) ** 2 / (v_c**2 / (len(c) - 1) + v_t**2 / (len(t) - 1)))
    t_crit = float(st.t.ppf(1 - alpha / 2, df))
    diff = float(m_t - m_c)
    stat = diff / se if se > 0 else 0.0
    out = {
        "control": float(m_c), "treatment": float(m_t), "n_control": len(c),
        "n_treatment": len(t), "diff": diff, "se": se,
        "ci": [diff - t_crit * se, diff + t_crit * se], "t": stat, "df": df,
        "p_value": float(2 * st.t.sf(abs(stat), df)), "relative": None, "relative_ci": None,
    }
    if m_c > 0 and m_t > 0:
        log_ratio = np.log(m_t / m_c)
        se_log = np.sqrt(v_t / m_t**2 + v_c / m_c**2)
        out["relative"] = float(np.expm1(log_ratio))
        out["relative_ci"] = [float(np.expm1(log_ratio - t_crit * se_log)),
                              float(np.expm1(log_ratio + t_crit * se_log))]
    return out


# ---------------------------------------------------------------- power and design
def power_two_proportions(p_control: float, lift: float, n_control: float, n_treatment: float,
                          alpha: float = 0.05) -> float:
    """Power of the two-sided unpooled z-test when the true absolute lift is ``lift``."""
    p_t = p_control + lift
    if not 0 < p_t < 1:
        raise ValueError("control rate + lift must lie in (0, 1)")
    se = np.sqrt(p_control * (1 - p_control) / n_control + p_t * (1 - p_t) / n_treatment)
    z = z_critical(alpha)
    shift = abs(lift) / se
    return float(st.norm.cdf(shift - z) + st.norm.cdf(-shift - z))


def minimum_detectable_effect(p_control: float, n_control: float, n_treatment: float,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest positive absolute lift detected with the target power (exact root of the power
    function, so ``power_two_proportions(p, mde, ...) == power``)."""
    upper = 1 - p_control - 1e-9
    if power_two_proportions(p_control, upper, n_control, n_treatment, alpha) < power:
        return float("inf")
    return float(optimize.brentq(
        lambda d: power_two_proportions(p_control, d, n_control, n_treatment, alpha) - power,
        1e-9, upper, xtol=1e-10))


def sample_size_per_arm(p_control: float, lift: float, alpha: float = 0.05,
                        power: float = 0.8) -> int:
    """Units per arm (equal allocation) for the unpooled z-test to reach ``power``."""
    p_t = p_control + lift
    var = p_control * (1 - p_control) + p_t * (1 - p_t)
    z_sum = z_critical(alpha) + st.norm.ppf(power)
    return int(np.ceil(z_sum**2 * var / lift**2))


def retrodesign(true_effect: float, se: float, alpha: float = 0.05) -> dict:
    """Power, sign-error rate and expected exaggeration of *significant* estimates when the
    true effect is ``true_effect`` and the estimator is N(true_effect, se^2) (Gelman & Carlin,
    2014). An exaggeration ratio of 1.6 means significant results overstate the effect by 60%
    on average."""
    d = abs(true_effect)
    c = z_critical(alpha) * se
    a, b = (c - d) / se, (-c - d) / se
    p_hi, p_lo = st.norm.sf(a), st.norm.cdf(b)
    power = p_hi + p_lo
    # E[|estimate| ; significant] from the truncated-normal first moments of both tails.
    abs_mass = (d * p_hi + se * st.norm.pdf(a)) + (-d * p_lo + se * st.norm.pdf(b))
    return {"true_effect": true_effect, "se": se, "power": float(power),
            "type_s_error": float(p_lo / power) if power > 0 else None,
            "exaggeration_ratio": float(abs_mass / power / d) if d > 0 else None}


# ---------------------------------------------------------------- multiplicity and heterogeneity
def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (controls the false discovery rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def cochran_q(estimates: Sequence[float], ses: Sequence[float]) -> dict:
    """Test that independent subgroup estimates share one effect (inverse-variance weights)."""
    est = np.asarray(estimates, dtype=float)
    w = 1 / np.asarray(ses, dtype=float) ** 2
    pooled = float((w * est).sum() / w.sum())
    q = float((w * (est - pooled) ** 2).sum())
    df = len(est) - 1
    return {"q": q, "df": df, "p_value": float(st.chi2.sf(q, df)), "pooled_effect": pooled}


# ---------------------------------------------------------------- design checks
def sample_ratio_test(counts: Sequence[int], expected_shares: Sequence[float]) -> dict:
    """Chi-square goodness of fit of arm sizes to the designed allocation (SRM check)."""
    obs = np.asarray(counts, dtype=float)
    exp = obs.sum() * np.asarray(expected_shares, dtype=float)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    return {"chi2": chi2, "p_value": float(st.chi2.sf(chi2, len(obs) - 1)),
            "observed_share": (obs / obs.sum()).tolist()}


def standardized_mean_difference(control: np.ndarray, treatment: np.ndarray) -> float:
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    pooled_sd = np.sqrt((c.var(ddof=1) + t.var(ddof=1)) / 2)
    return 0.0 if pooled_sd == 0 else float((t.mean() - c.mean()) / pooled_sd)


def permutation_p_value(outcome: np.ndarray, treated: np.ndarray, n_permutations: int,
                        seed: int) -> float:
    """Two-sided randomization-inference p-value for the difference in means, re-drawing the
    assignment vector with the observed arm sizes."""
    y = np.asarray(outcome, dtype=float)
    z = np.asarray(treated, dtype=bool)
    n, n_t = len(y), int(z.sum())
    observed = abs(y[z].mean() - y[~z].mean())
    rng = np.random.default_rng(seed)
    total = y.sum()
    extreme = 0
    for start in range(0, n_permutations, 1000):
        size = min(1000, n_permutations - start)
        keys = rng.random((size, n))
        # Rank-based draw of exactly n_t treated units per permutation.
        mask = np.argsort(keys, axis=1)[:, :n_t]
        s_t = y[mask].sum(axis=1)
        diff = s_t / n_t - (total - s_t) / (n - n_t)
        extreme += int((np.abs(diff) >= observed - 1e-12).sum())
    return (extreme + 1) / (n_permutations + 1)


def aa_simulation(outcome: np.ndarray, n_simulations: int, seed: int, alpha: float = 0.05,
                  treatment_share: float = 0.5) -> dict:
    """Randomly split units with no true effect many times and record how often the
    two-proportion test rejects and how often its CI covers zero."""
    y = np.asarray(outcome, dtype=float)
    rng = np.random.default_rng(seed)
    rejections = covered = 0
    for _ in range(n_simulations):
        z = rng.random(len(y)) < treatment_share
        res = two_proportion_test(int(y[~z].sum()), int((~z).sum()), int(y[z].sum()),
                                  int(z.sum()), alpha)
        rejections += res["p_value"] < alpha
        covered += res["ci"][0] <= 0 <= res["ci"][1]
    rate = rejections / n_simulations
    half = z_critical(0.05) * np.sqrt(rate * (1 - rate) / n_simulations)
    return {"simulations": n_simulations, "units": len(y), "nominal_alpha": alpha,
            "false_positive_rate": rate, "false_positive_rate_ci": [rate - half, rate + half],
            "ci_coverage_of_zero": covered / n_simulations}


def regression_adjusted_effect(y: np.ndarray, treated: np.ndarray, covariates: np.ndarray | None,
                               alpha: float = 0.05) -> dict:
    """Lin (2013) regression adjustment: OLS of y on treatment, centered pre-treatment covariates
    and their interactions with treatment, with HC2 robust standard errors. Unbiased for the
    average treatment effect under randomization and never less precise asymptotically."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(treated, dtype=float)
    n = len(y)
    parts = [np.ones(n), z]
    n_covariates = 0
    if covariates is not None and covariates.size:
        x = np.asarray(covariates, dtype=float)
        x = x - x.mean(axis=0)
        parts += [x, x * z[:, None]]
        n_covariates = x.shape[1]
    design = np.column_stack(parts)
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    meat = (design * (resid**2 / np.clip(1 - leverage, 1e-12, None))[:, None]).T @ design
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[1, 1]))
    est = float(beta[1])
    z_crit = z_critical(alpha)
    stat = est / se if se > 0 else 0.0
    return {"diff": est, "se": se, "ci": [est - z_crit * se, est + z_crit * se],
            "p_value": float(2 * st.norm.sf(abs(stat))), "n_covariates": n_covariates}
