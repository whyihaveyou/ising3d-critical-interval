"""Check the five active infrared integrals by exact cubic lattice Green values.

The analytic route uses only rational interval arithmetic. The optional comparison
reads the original angular-walk bounds from the project's production generator.
"""

import argparse
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import json
from math import comb, floor, ceil, isqrt, prod
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY = ("STAGE21_ORDERED_780_493.json", "STAGE20_ORDERED_price_791_500.json")
GRID = 10**18
ORIGIN = (0, 0, 0)
MOMENTS = {
    # Coefficients of (g0, 1/(pi^2 g0), 1) in G(3;l,m,n).
    # Glasser and Boersma, J. Phys. A 33 (2000), 5020, Table 1.
    (0, 0, 0): (1, 0, 0),
    (1, 0, 0): (1, 0, -F(1, 3)),
    (1, 1, 0): (F(5, 12), -F(1, 2), 0),
    (2, 0, 0): (F(10, 3), 2, -2),
    (1, 1, 1): (-F(1, 8), F(3, 4), 0),
    (2, 1, 0): (F(3, 8), -F(9, 4), F(1, 3)),
    (3, 0, 0): (F(35, 2), 21, -13),
    (2, 1, 1): (-F(2, 3), 2, 0),
    (2, 2, 0): (F(73, 36), -F(29, 6), 0),
    (3, 1, 0): (-F(79, 36), -F(85, 6), 4),
    (4, 0, 0): (F(994, 9), F(542, 3), -92),
}
INPUTS = {
    29: (2, [2126, 2110, -119, -119]),
    32: (0, [-1653, -613, -613, -613, 609, 1159, 609, 1159, 1159, 609]),
    33: (6, [-1722, -1421, -1417, -1417]),
    35: (3, [-1772, -1387, -1479, -1322]),
    69: (2, [2090, 2101, -330, -330]),
}


def canonical(key):
    return tuple(sorted(map(abs, key), reverse=True))


def check_harmonic():
    for x, row in MOMENTS.items():
        if sum(x) > 3:
            continue
        adjacent = []
        for j in range(3):
            for d in (-1, 1):
                v = list(x)
                v[j] += d
                adjacent.append(MOMENTS[canonical(v)])
        difference = tuple(6*r - sum(col) for r, col in zip(row, zip(*adjacent)))
        assert difference == (0, 0, 2 * (x == ORIGIN))


def square_root(n):
    den = 10**40
    a = isqrt(n*den**2)
    return F(a, den), F(a+1, den)


def atan_reciprocal(q):
    # An even number of terms is a lower endpoint for positive 1/q.
    terms = 30
    lo = sum(F((-1)**j, (2*j+1)*q**(2*j+1)) for j in range(terms))
    return lo, lo+F(1, (2*terms+1)*q**(2*terms+1))


def analytic_constants():
    two, three, six = map(square_root, (2, 3, 6))
    k_lo = (2-three[1])*(three[0]-two[1])
    k_hi = (2-three[0])*(three[1]-two[0])
    base_lo = 18+12*two[0]-10*three[1]-7*six[1]
    base_hi = 18+12*two[1]-10*three[0]-7*six[0]
    assert 0 < base_lo < base_hi and 0 < k_lo < k_hi < F(1, 10)
    # Watson's formula includes (2K(k_minus)/pi)^2.
    terms = 18
    coefficients = [F(comb(2*j, j)**2, 16**j) for j in range(terms)]
    series = lambda k: sum(c*k**(2*j) for j,c in enumerate(coefficients))
    E_lo = series(k_lo)
    E_hi = series(k_hi) + k_hi**(2*terms)/(1-k_hi**2)
    # Compress large exact intermediate fractions to readily inspectable bounds.
    den = 10**32
    g_lo = F(floor(den*base_lo*E_lo**2), den)
    g_hi = F(ceil(den*base_hi*E_hi**2), den)
    a, b = atan_reciprocal(5), atan_reciprocal(239)
    pi_lo, pi_hi = 16*a[0]-4*b[1], 16*a[1]-4*b[0]
    assert 3 < pi_lo < pi_hi < 4 and g_hi-g_lo <= F(2, den)
    H_lo = F(floor(den/(pi_hi**2*g_hi)), den)
    H_hi = F(ceil(den/(pi_lo**2*g_lo)), den)
    return ((g_lo, g_hi), (H_lo, H_hi), (F(1), F(1)))


def product_poly(a, b):
    out = defaultdict(F)
    for ia,x in a.items():
        for ib,y in b.items():
            out[tuple(u+v for u,v in zip(ia,ib))] += x*y
    return dict(out)


def symmetric_polynomial(tag, root):
    basis = sorted((x for x in product(range(3), repeat=3) if sum(x)<=2),
                   key=lambda x: (sum(x), x))
    poly = dict(zip(basis, root))
    one = {ORIGIN:1}
    x = (1, 0, 0)
    y = (0, 1, 0)
    factors = {0: [], 2: [(x,1),(y,1)], 3: [(x,1),(y,-1)], 6: [(x,-1),(y,-1)]}
    square = product_poly(poly, poly)
    for axis, sign in factors[tag]:
        square = product_poly(square, {**one, axis:sign})
    symmetric = defaultdict(F)
    for key,x in square.items():
        symmetric[canonical(key)] += x
    norm = sum(map(abs, symmetric.values()))
    assert norm > 0
    return {key:x/norm for key,x in symmetric.items() if x}


def harmonic_coefficients(poly):
    """Integral P/lambda; lambda=2*(3-cos(k1)-cos(k2)-cos(k3))."""
    coeff = [F(0) for _ in range(3)]
    for key, c in poly.items():
        for indices in product(*(range(p+1) for p in key)):
            freq = canonical(p-2*i for p,i in zip(key,indices))
            weight = c * prod(comb(p,i) for p,i in zip(key,indices)) / 2**(1+sum(key))
            for j, r in enumerate(MOMENTS[freq]):
                coeff[j] += weight*r
    return coeff


def integral_bounds(poly, constants):
    coeff = harmonic_coefficients(poly)
    lo = sum(c * k[0 if c>=0 else 1] for c,k in zip(coeff,constants))
    hi = sum(c * k[1 if c>=0 else 0] for c,k in zip(coeff,constants))
    lo = F(floor(GRID*lo),GRID)
    hi = F(ceil(GRID*hi),GRID)
    assert hi-lo == F(1,GRID) and lo>0
    return lo, hi


def archived_comparison(polys, bounds):
    import STAGE21_ORDERED_active_rows as original

    phase = original.current.stage.W.fixed.phase
    green = phase.AngularGreen(F(0))
    old_bounds = {}
    for j, poly in polys.items():
        mapped = [poly.get(canonical(key),0) for key in phase.KEYS]
        assert mapped == phase.polynomial(*INPUTS[j])
        old = green.upper(mapped)/6
        assert bounds[j][1] < old
        old_bounds[j] = old
        print(f"row={j} 1e18*J in [{GRID*bounds[j][0]}, {GRID*bounds[j][1]}], "
              f"floor(1e15*(Jold-Jhi))={floor(10**15*(old-bounds[j][1]))}")
    gains = []
    for archive in LEGACY:
        src = json.loads((HERE/archive).read_text())
        rows = dict(src["rows"]["ordinary"])
        pairs = original.active_tests(src["rows"]["ordinary"]).items()
        assert {i:x for i,x in pairs if x[0]!="extension"} == INPUTS
        assert set(rows)==set(INPUTS)|{42,45,76,83}
        divisor = 10**6*src["denominator"]
        lo = sum(rows[i]*(old_bounds[i]-bounds[i][1]) for i in INPUTS)/divisor
        hi = sum(rows[i]*(old_bounds[i]-bounds[i][0]) for i in INPUTS)/divisor
        assert 0<lo<=hi
        gains.append((lo,hi))
        print(f"{archive} 1e18*weighted Green improvement in "
              f"[{floor(GRID*lo)}, {ceil(GRID*hi)}]")
        for r in (F(791,500), F(2003,1266)):
            c_min = 4/(r**2-1)
            c_max = c_min+4
            print(f"r={r}: fieldwise improvement 1e15 bounds "
                  f"[{floor(10**15*c_min*lo)}, {ceil(10**15*c_max*hi)}]")
    new = json.loads((HERE/"STAGE22_ORDERED_HOLDBACK_2003_1266.json").read_text())
    r, q = F(new["r"]), F(new["Q"])
    refined = F(new["L"])+4*gains[0][0]/((r**2-1)*q)
    print("floor(1e15*m2_certified_at_2003/1266) =",floor(10**15*refined), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-archive", action="store_true")
    args = parser.parse_args()
    assert __debug__
    check_harmonic()
    constants = analytic_constants()
    polys = {j:symmetric_polynomial(*pair) for j,pair in INPUTS.items()}
    bounds = {j:integral_bounds(poly,constants) for j,poly in polys.items()}
    print("PASS 11 analytic Green inputs, 7 displacement recurrence checks, "
          "5 normalized spectral integrals with interval width 1e-18", flush=True)
    print("Green G(3;0,0,0) x 1e32 bounds", *(int(z*10**32) for z in constants[0]))
    for j,(lo,hi) in bounds.items():
        print(f"J[{j}] x 1e18 bounds", int(lo*GRID), int(hi*GRID))
    if args.compare_archive:
        archived_comparison(polys,bounds)
