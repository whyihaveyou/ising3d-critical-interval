"""Integer walk sum and rational tail bounds for the zero-mode Green function."""
from fractions import Fraction as F
from math import comb


def watson_bounds():
    """Integral of 1/(1-gamma). W_N counts even returns, with a proved upper tail."""
    order = 400
    scaled_sum = 1
    for n in range(1, order + 1):
        term = total = 1
        for k in range(n):
            term, remainder = divmod(2*(2*k+1)*(n-k)**2*term, (k+1)**3)
            assert remainder == 0
            total += term
        scaled_sum = 36*scaled_sum + comb(2*n, n)*total
    lower = F(scaled_sum, 36**order)
    coefficient, outer = F(241, 1000), F(1105, 1152)
    assert (4*coefficient)**2 > (F(144, 47)/F(157, 50))**3
    upper = lower + 2*coefficient/20 + 2*outer**(2*order+2)/(1-outer**2)
    assert F(1493078, 10**6) < lower and upper < F(1517180, 10**6)
    return lower, upper


def improve_watson():
    """Bound the even return probabilities beyond the exact first 400 terms.

    Write gamma = sum(cos(k_i))/3 and double its nonnegative half-zone. On the
    unit coordinate cube, Taylor's formula and exp(t)<=1+t*exp(t) give
    gamma^(2n) <= exp(-n*|k|^2/3)+(n/36)*sum(k_i^4)*exp(-11*n*|k|^2/36).
    Outside, positive gamma <= 61/72. Integrating the two Gaussians gives
    p_(2n) <= a*n^(-3/2)+b*n^(-5/2)+2*(61/72)^(2n), with
    a=(3/pi)^(3/2)/4 and b=(pi*c)^(-3/2)/(64*c^2), c=11/36.

    On the ball |k|<=sqrt(6), gamma>=1-|k|^2/6. Its positive radial integral
    is 3*sqrt(6)/(2*pi^(3/2))*Gamma(2n+1)/Gamma(2n+5/2) after doubling.
    Gamma log-convexity gives Gamma(x+3/2) <= Gamma(x)*x*sqrt(x+1),
    hence p_(2n) >= a*(n+1)^(-3/2). Compare both tails to integrals.
    """
    w400, _ = watson_bounds()
    # Machin's identity with rational alternating-series enclosures for pi.
    def atan_bounds(x):
        partial = sum((-1)**j*x**(2*j+1)/(2*j+1) for j in range(6))
        return partial, partial+x**13/13
    l5, h5 = atan_bounds(F(1, 5))
    l239, h239 = atan_bounds(F(1, 239))
    pl, pu = 16*l5-4*h239, 16*h5-4*l239
    al, au, bu, c = F(23329, 100000), F(23330, 100000), F(19, 100), F(11, 36)
    assert al**2 < 27/(16*pu**3) and au**2 > 27/(16*pl**3)
    assert bu**2 > 1/(64**2*c**4*(pl*c)**3)
    outer = F(61, 72)
    # a/sqrt(402) uses the upward square-root bound 20.05.
    assert F(401, 20)**2 > 402
    lo = w400+2*al/F(401, 20)
    hi = w400+2*au/20+2*bu/(3*20**3)+2*outer**802/(1-outer**2)
    assert lo < hi
    return lo, hi
