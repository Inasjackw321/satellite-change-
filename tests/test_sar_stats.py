import numpy as np
import pytest

from sar_stats import DEFAULT_ENL, LRT_EXPRESSION, chi2_threshold

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
    L1, L2 = DEFAULT_ENL * n1, DEFAULT_ENL * n2
    stat = sum(
        lrt(speckle(rng, mean, L1), speckle(rng, mean, L2), L1, L2)
        for mean in (0.2, 0.04)  # VV, VH
    )
    rate = np.mean(stat > chi2_threshold(alpha, dof=2))
    assert rate == pytest.approx(alpha, rel=0.05)


def test_detects_a_5db_change_with_four_images_per_window():
    rng = np.random.default_rng(2)
    L = DEFAULT_ENL * 4
    stat = sum(
        lrt(speckle(rng, mean, L), speckle(rng, 10**0.5 * mean, L), L, L)
        for mean in (0.2, 0.04)
    )
    assert np.mean(stat > chi2_threshold(0.01, dof=2)) > 0.95


def test_threshold_rejects_bad_alpha():
    with pytest.raises(ValueError):
        chi2_threshold(0, 2)
