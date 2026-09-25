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

The statistic is written once as an expression string so that Earth Engine
(``ee.Image.expression``) and NumPy (in the tests) evaluate exactly the
same formula.
"""

from scipy import stats

# ESA's nominal equivalent number of looks for Sentinel-1 IW GRDH products.
DEFAULT_ENL = 4.4

# Bartlett-corrected -2 log Q for a single band. s1, s2 are mean intensities
# (linear power, not dB) and L1, L2 their numbers of looks. Under H0 the
# common mean is the looks-weighted average of s1 and s2.
LRT_EXPRESSION = (
    "2 * ((L1 + L2) * log((L1 * s1 + L2 * s2) / (L1 + L2))"
    " - L1 * log(s1) - L2 * log(s2))"
    " / (1 + (1 / L1 + 1 / L2 - 1 / (L1 + L2)) / 6)"
)


def chi2_threshold(alpha, dof):
    """Critical value of -2 log Q at significance level ``alpha``.

    A pixel whose statistic exceeds this value is flagged as changed. The
    expected false alarm rate on unchanged pixels is ``alpha``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(stats.chi2.ppf(1 - alpha, dof))

