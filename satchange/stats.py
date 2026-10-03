"""Statistics for Sentinel-1 change detection.

Follows "Detecting Changes in Sentinel-1 Imagery (Part 1)" by Mort Canty
(Google Earth Engine community tutorials):

* A multilook SAR intensity pixel with mean ``a`` and ``L`` looks is
  gamma-distributed:  s ~ Gamma(shape=L, scale=a/L).
* To test whether two acquisitions share the same mean (no change), use the
  likelihood ratio test. Under the null hypothesis ``-2 log Q`` is
  approximately chi-square distributed with one degree of freedom per band.
  Adding the VV and VH statistics gives the bivariate test with 2 dof.

The tutorial compares two single acquisitions with the same number of looks
``m``. Here each side can be the average of several acquisitions, so each
side has its own number of looks (``L = m * n_images``). With
``L1 == L2 == m`` this reduces to the tutorial's

    Q = 2^(2m) * (s1 * s2)^m / (s1 + s2)^(2m).

The chi-square approximation is loose at the ~4 looks of a single
Sentinel-1 GRD scene (about 25% more false alarms than ``alpha``). This
test is Bartlett's test for equal variances (a gamma variable with L looks is
a scaled chi-square with 2L dof), so we divide by Bartlett's correction
factor, which brings the false alarm rate back to ``alpha``.

Everything is computed locally with NumPy at full 10 m resolution.
"""

import numpy as np
from scipy import optimize, stats

# ESA's nominal equivalent number of looks for Sentinel-1 IW GRDH products.
# Terrain-corrected products can differ; see estimate_enl.
NOMINAL_ENL = 4.4


def lrt(s1, s2, L1, L2):
    """Bartlett-corrected -2 log Q for one band.

    s1, s2 are mean intensities (linear power, not dB) and L1, L2 their
    numbers of looks. Under H0 the common mean is the looks-weighted average
    of s1 and s2. Works on scalars and arrays.
    """
    s1, s2, L1, L2 = (np.asarray(v, dtype=np.float64) for v in (s1, s2, L1, L2))
    pooled = (L1 * s1 + L2 * s2) / (L1 + L2)
    m2logq = 2 * ((L1 + L2) * np.log(pooled) - L1 * np.log(s1) - L2 * np.log(s2))
    bartlett = 1 + (1 / L1 + 1 / L2 - 1 / (L1 + L2)) / 6
    return np.maximum(m2logq / bartlett, 0)


# The statistic is stored as int16: round(stat * STAT_SCALE), signed by
# the direction of change (+ increase, - decrease). Values are capped at
# STAT_MAX, far beyond any useful threshold (chi2 at alpha=1e-6, 2 dof ~ 27.6).
STAT_SCALE = 100
STAT_MAX = 320
NODATA = -32768


def chi2_threshold(alpha, dof):
    """Critical value of -2 log Q at significance level ``alpha``.

    A pixel whose statistic exceeds this value is flagged as changed. The
    expected false alarm rate on unchanged pixels is ``alpha``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(stats.chi2.ppf(1 - alpha, dof))


def log_ratio_iqr(looks):
    """Interquartile range of log(s_a / s_b) for two unchanged scenes.

    With the same mean, s_a / s_b ~ F(2L, 2L), whose log is symmetric about 0.
    """
    return 2 * np.log(stats.f.ppf(0.75, 2 * looks, 2 * looks))


def estimate_enl(log_ratios, bounds=(0.5, 200.0)):
    """Estimate the equivalent number of looks of single scenes.

    ``log_ratios`` are log(s_a / s_b) for pixels of two acquisitions that
    should mostly be unchanged (e.g. 12 days apart). The sample IQR is
    matched to ``log_ratio_iqr``. This ignores a global calibration offset
    between the dates, and pixels that did change only widen the spread,
    biasing the ENL low (5% of pixels changed: ~10% low). A low ENL makes
    the test stricter, never looser.
    """
    x = np.asarray(log_ratios, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError("need at least 100 valid samples to estimate the ENL")
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    lo, hi = bounds
    # log_ratio_iqr decreases with looks; clamp outside the bracket.
    if iqr >= log_ratio_iqr(lo):
        return lo
    if iqr <= log_ratio_iqr(hi):
        return hi
    return float(optimize.brentq(lambda L: log_ratio_iqr(L) - iqr, lo, hi))


def decode(signed):
    """Split the stored int16 array into (statistic, increased, valid)."""
    valid = signed != NODATA
    stat = np.abs(signed.astype(np.float32)) / STAT_SCALE
    return stat, signed > 0, valid
