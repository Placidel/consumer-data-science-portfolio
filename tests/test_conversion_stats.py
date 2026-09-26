"""Section 03 statistics: agreement with scipy, test/CI duality, calibration and design math."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as st

from northstar.conversion import stats


def test_two_proportion_test_matches_hand_calculation_and_chi_square():
    r = stats.two_proportion_test(200, 1000, 250, 1000)
    se = np.sqrt(0.2 * 0.8 / 1000 + 0.25 * 0.75 / 1000)
    assert r["diff"] == pytest.approx(0.05)
    assert r["se"] == pytest.approx(se)
    assert r["ci"] == pytest.approx([0.05 - 1.959964 * se, 0.05 + 1.959964 * se], rel=1e-5)
    assert r["p_value"] == pytest.approx(2 * st.norm.sf(0.05 / se))
    # The pooled test equals Pearson's chi-square without continuity correction.
    chi2 = st.chi2_contingency([[200, 800], [250, 750]], correction=False)
    assert r["p_value_pooled"] == pytest.approx(chi2.pvalue)
    assert r["relative"] == pytest.approx(0.25)
    lo, hi = r["relative_ci"]
    assert lo < 0.25 < hi


@pytest.mark.parametrize("seed", range(40))
def test_p_value_and_confidence_interval_always_agree(seed):
    rng = np.random.default_rng(seed)
    n_c, n_t = rng.integers(50, 3000, size=2)
    x_c, x_t = rng.binomial(n_c, 0.3), rng.binomial(n_t, rng.uniform(0.25, 0.4))
    for alpha in (0.05, 0.01, 0.1):
        r = stats.two_proportion_test(int(x_c), int(n_c), int(x_t), int(n_t), alpha)
        assert (r["p_value"] < alpha) == (r["ci"][0] > 0 or r["ci"][1] < 0)
    a, b = rng.lognormal(3, 1, n_c), rng.lognormal(3.05, 1, n_t)
    w = stats.welch_test(a, b)
    assert (w["p_value"] < 0.05) == (w["ci"][0] > 0 or w["ci"][1] < 0)


def test_confidence_interval_covers_the_true_difference_at_the_nominal_rate():
    rng = np.random.default_rng(0)
    n, p_c, p_t, sims = 2000, 0.24, 0.27, 3000
    x_c, x_t = rng.binomial(n, p_c, sims), rng.binomial(n, p_t, sims)
    covered = [stats.two_proportion_test(int(a), n, int(b), n)["ci"] for a, b in
               zip(x_c, x_t, strict=True)]
    rate = np.mean([lo <= p_t - p_c <= hi for lo, hi in covered])
    assert abs(rate - 0.95) < 0.015


def test_welch_test_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.gamma(2, 40, 400), rng.gamma(2, 36, 450)
    r = stats.welch_test(a, b)
    ref = st.ttest_ind(b, a, equal_var=False)
    assert r["p_value"] == pytest.approx(ref.pvalue)
    ci = ref.confidence_interval(0.95)
    assert r["ci"] == pytest.approx([ci.low, ci.high])
    assert r["relative"] == pytest.approx(b.mean() / a.mean() - 1)


def test_power_matches_monte_carlo_and_mde_inverts_power():
    p0, lift, n = 0.25, 0.03, 3000
    analytic = stats.power_two_proportions(p0, lift, n, n)
    rng = np.random.default_rng(2)
    sims = 4000
    x_c, x_t = rng.binomial(n, p0, sims), rng.binomial(n, p0 + lift, sims)
    empirical = np.mean([stats.two_proportion_test(int(a), n, int(b), n)["p_value"] < 0.05
                         for a, b in zip(x_c, x_t, strict=True)])
    assert abs(empirical - analytic) < 0.03
    assert stats.power_two_proportions(p0, 0.0, n, n) == pytest.approx(0.05)
    mde = stats.minimum_detectable_effect(p0, n, n, 0.05, 0.8)
    assert stats.power_two_proportions(p0, mde, n, n) == pytest.approx(0.8, abs=1e-6)
    # More units -> smaller MDE; the textbook 2.8-sigma approximation is close.
    assert stats.minimum_detectable_effect(p0, 4 * n, 4 * n) < mde
    assert mde == pytest.approx(2.8 * np.sqrt(2 * p0 * (1 - p0) / n), rel=0.05)


def test_sample_size_reaches_target_power():
    n = stats.sample_size_per_arm(0.25, 0.02, 0.05, 0.8)
    assert stats.power_two_proportions(0.25, 0.02, n, n) >= 0.8
    assert stats.power_two_proportions(0.25, 0.02, n * 0.95, n * 0.95) < 0.8


def test_retrodesign_exaggeration_shrinks_with_power():
    weak = stats.retrodesign(0.02, 0.0115)
    strong = stats.retrodesign(0.02, 0.004)
    assert weak["power"] == pytest.approx(
        st.norm.sf(1.959964 - 0.02 / 0.0115) + st.norm.cdf(-1.959964 - 0.02 / 0.0115), rel=1e-4)
    assert weak["exaggeration_ratio"] > 1.3 > strong["exaggeration_ratio"] >= 1.0
    # Monte Carlo check of E[|estimate| | significant] / true effect.
    rng = np.random.default_rng(3)
    est = rng.normal(0.02, 0.0115, 400_000)
    sig = np.abs(est) > 1.959964 * 0.0115
    assert np.abs(est[sig]).mean() / 0.02 == pytest.approx(weak["exaggeration_ratio"], rel=0.01)
    assert (est[sig] < 0).mean() == pytest.approx(weak["type_s_error"], abs=0.002)


def test_holm_and_benjamini_hochberg_on_a_worked_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # Holm: sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 with running max.
    assert stats.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # BH: sorted p * m / rank -> 0.02, 0.02, 0.04, 0.04 with running min from the top.
    assert stats.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert (stats.holm(p) >= stats.benjamini_hochberg(p)).all()
    assert stats.holm([0.9, 0.8]).max() == 1.0


def test_cochran_q_detects_heterogeneity_only_when_present():
    same = stats.cochran_q([0.03, 0.03, 0.03], [0.01, 0.02, 0.015])
    assert same["q"] == pytest.approx(0.0) and same["p_value"] == pytest.approx(1.0)
    assert same["pooled_effect"] == pytest.approx(0.03)
    different = stats.cochran_q([-0.05, 0.0, 0.08], [0.01, 0.01, 0.01])
    assert different["p_value"] < 1e-6 and different["df"] == 2


def test_sample_ratio_mismatch_check():
    assert stats.sample_ratio_test([5000, 5000], [0.5, 0.5])["p_value"] == pytest.approx(1.0)
    assert stats.sample_ratio_test([5000, 5400], [0.5, 0.5])["p_value"] < 0.001
    assert stats.sample_ratio_test([2000, 8000], [0.2, 0.8])["p_value"] == pytest.approx(1.0)


def test_permutation_p_value_agrees_with_the_z_test_and_is_null_calibrated():
    rng = np.random.default_rng(4)
    z = rng.random(3000) < 0.5
    y = rng.random(3000) < np.where(z, 0.29, 0.25)
    perm = stats.permutation_p_value(y, z, 4000, seed=1)
    ztest = stats.two_proportion_test(int(y[~z].sum()), int((~z).sum()), int(y[z].sum()),
                                      int(z.sum()))
    assert perm == pytest.approx(ztest["p_value"], abs=0.01)
    null = stats.permutation_p_value(rng.random(3000) < 0.25, z, 2000, seed=2)
    assert 0 < null <= 1


def test_aa_simulation_is_calibrated():
    y = np.random.default_rng(5).random(3000) < 0.25
    aa = stats.aa_simulation(y, 1500, seed=7)
    assert abs(aa["false_positive_rate"] - 0.05) < 0.02
    assert aa["ci_coverage_of_zero"] == pytest.approx(1 - aa["false_positive_rate"])


def test_regression_adjustment_is_unbiased_and_gains_precision_from_a_predictive_covariate():
    rng = np.random.default_rng(6)
    n = 6000
    x = rng.normal(size=n)
    z = rng.random(n) < 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 1.2 * x + 0.2 * z)))).astype(float)
    plain = stats.regression_adjusted_effect(y, z, None)
    diff = y[z].mean() - y[~z].mean()
    assert plain["diff"] == pytest.approx(diff)
    assert plain["n_covariates"] == 0
    adjusted = stats.regression_adjusted_effect(y, z, x[:, None])
    assert adjusted["n_covariates"] == 1
    assert adjusted["se"] < 0.9 * plain["se"]
    assert abs(adjusted["diff"] - plain["diff"]) < 2 * plain["se"]


def test_standardized_mean_difference():
    assert stats.standardized_mean_difference(np.zeros(5), np.zeros(5)) == 0.0
    a, b = np.array([0.0, 1, 2, 3]), np.array([1.0, 2, 3, 4])
    assert stats.standardized_mean_difference(a, b) == pytest.approx(1 / np.std(a, ddof=1))
