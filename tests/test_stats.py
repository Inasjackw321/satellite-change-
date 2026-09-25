import numpy as np
import pytest

from satchange.stats import (
    LRT_EXPRESSION, NODATA, NOMINAL_ENL, chi2_threshold, decode, estimate_enl, exceedance,
)

N = 400_000


def lrt(s1, s2, L1, L2):
    """Evaluate the same expression string Earth Engine uses."""
    return eval(LRT_EXPRESSION, {"log": np.log}, {"s1": s1, "s2": s2, "L1": L1, "L2": L2})


def speckle(rng, mean, looks, n=N):
    """Multilook intensity: Gamma(shape=looks, scale=mean/looks)."""
    return rng.gamma(looks, mean / looks, n)


def test_reduces_to_tutorial_formula_for_equal_looks():
    rng = np.random.default_rng(0)
    m = 5
    s1, s2 = speckle(rng, 0.1, m, 1000), speckle(rng, 0.3, m, 1000)
    q = 2 ** (2 * m) * (s1 * s2) ** m / (s1 + s2) ** (2 * m)
    bartlett = 1 + 1 / (4 * m)
    np.testing.assert_allclose(lrt(s1, s2, m, m), -2 * np.log(q) / bartlett, rtol=1e-9)


@pytest.mark.parametrize("n1,n2", [(1, 1), (4, 3), (8, 2)])
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_false_alarm_rate_matches_alpha_bivariate(n1, n2, alpha):
    """No change: VV and VH with the same mean before and after."""
    rng = np.random.default_rng(1)
    L1, L2 = NOMINAL_ENL * n1, NOMINAL_ENL * n2
    stat = sum(
        lrt(speckle(rng, mean, L1), speckle(rng, mean, L2), L1, L2)
        for mean in (0.2, 0.04)  # VV, VH
    )
    rate = np.mean(stat > chi2_threshold(alpha, dof=2))
    assert rate == pytest.approx(alpha, rel=0.05)


def test_detects_a_5db_change_with_four_images_per_window():
    rng = np.random.default_rng(2)
    L = NOMINAL_ENL * 4
    stat = sum(
        lrt(speckle(rng, mean, L), speckle(rng, 10**0.5 * mean, L), L, L)
        for mean in (0.2, 0.04)
    )
    assert np.mean(stat > chi2_threshold(0.01, dof=2)) > 0.95


def test_threshold_rejects_bad_alpha():
    with pytest.raises(ValueError):
        chi2_threshold(0, 2)


@pytest.mark.parametrize("looks", [2.0, 4.4, 7.5])
def test_estimate_enl_recovers_looks(looks):
    rng = np.random.default_rng(3)
    mean = rng.lognormal(-2, 1, 50_000)  # varied land cover
    log_ratio = np.log(speckle(rng, mean, looks, mean.size) / speckle(rng, mean, looks, mean.size))
    assert estimate_enl(log_ratio) == pytest.approx(looks, rel=0.05)


def test_estimate_enl_with_some_change_errs_conservative():
    """Real change widens the spread, so the ENL comes out low, which only
    makes the test stricter (fewer false alarms)."""
    rng = np.random.default_rng(4)
    n = 50_000
    a, b = speckle(rng, 0.1, 4.4, n), speckle(rng, 0.1, 4.4, n)
    b[: n // 20] *= 10  # 5% of pixels changed by 10 dB
    log_ratio = np.log(a / b) + 0.2  # plus a global offset between dates
    assert 0.85 * 4.4 < estimate_enl(log_ratio) <= 4.4


def test_exceedance_interpolates_within_bin():
    counts = np.array([10, 10, 10, 10])  # uniform on [0, 4) with bin width 1
    assert exceedance(counts, 1.0, 0.0) == pytest.approx(1.0)
    assert exceedance(counts, 1.0, 2.5) == pytest.approx(0.375)
    assert exceedance(counts, 1.0, 9.0) == 0.0


def test_exceedance_of_simulated_null_matches_alpha():
    rng = np.random.default_rng(5)
    L = NOMINAL_ENL * 3
    stat = sum(lrt(speckle(rng, m, L), speckle(rng, m, L), L, L) for m in (0.2, 0.04))
    counts, _ = np.histogram(np.minimum(stat, 49.99), bins=500, range=(0, 50))
    assert exceedance(counts, 0.1, chi2_threshold(0.01, 2)) == pytest.approx(0.01, rel=0.05)


def test_decode():
    signed = np.array([[NODATA, 0, 1234, -950]], dtype=np.int16)
    stat, increased, valid = decode(signed)
    np.testing.assert_array_equal(valid, [[False, True, True, True]])
    np.testing.assert_allclose(stat[0, 1:], [0, 12.34, 9.5], rtol=1e-6)
    np.testing.assert_array_equal(increased[0, 1:], [False, True, False])
