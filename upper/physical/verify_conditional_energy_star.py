"""Certify local energies of the cubic Ising model by conditioning a two-step ball.

The linear solver only proposes multipliers. The certificate maximizes their
rounded residual over every boundary orbit, with an integer error bound.
"""
from fractions import Fraction
from collections import Counter
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, lcm

import numpy as np
from scipy.optimize import linprog


ORIGIN = (0, 0, 0)
DIRECTIONS = sorted(v for v in product(range(-1, 2), repeat=3) if sum(map(abs, v)) == 1)
INTERIOR = [ORIGIN] + DIRECTIONS
BOUNDARY = sorted(v for v in product(range(-2, 3), repeat=3) if sum(map(abs, v)) == 2)
SITES = INTERIOR + BOUNDARY
INDEX = {v: i for i, v in enumerate(SITES)}
SCALE = 1 << 22
GROUP = []
for axes in permutations(range(3)):
    for signs in product((-1, 1), repeat=3):
        GROUP.append(tuple(INDEX[tuple(s * p[j] for s, j in zip(signs, axes))] for p in SITES))
TOTAL_SCALE = len(GROUP) * SCALE


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def mask(points):
    return sum(1 << INDEX[p] for p in points)


def bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length()-1
        value -= bit


@lru_cache(None)
def orbit(pattern):
    sites = tuple(bits(pattern))
    return sorted({sum(1 << rotation[i] for i in sites) for rotation in GROUP})


@lru_cache(None)
def boundary_representatives():
    # Bit 1 is spin minus. A global flip preserves every even observable used below.
    count = 1 << len(BOUNDARY)
    seen = bytearray(count)
    representatives = []
    for word in range(count):
        if seen[word]:
            continue
        representatives.append(word)
        for full in orbit(word << len(INTERIOR)):
            image = full >> len(INTERIOR)
            seen[image] = seen[count-1-image] = 1
    assert all(seen)
    return np.asarray(representatives, dtype=np.int32)


def translation_rows(max_degree):
    # The two ends of each edge are the same lattice pattern seen at different origins.
    parents = {}

    def root(x):
        while parents.get(x, x) != x:
            parents[x] = parents.get(parents[x], parents[x])
            x = parents[x]
        return x

    for dx in range(1, 5):
        for dy in range(dx+1):
            for dz in range(dy+1):
                shift = (dx, dy, dz)
                overlap = [v for v in SITES if add(v, shift) in INDEX]
                for size in range(2, min(max_degree, len(overlap))+1, 2):
                    for pattern in combinations(overlap, size):
                        a = orbit(mask(pattern))[0]
                        b = orbit(mask(add(v, shift) for v in pattern))[0]
                        if a != b:
                            a, b = root(a), root(b)
                            parents[max(a, b)] = min(a, b)
    return [(a, root(a)) for a in sorted(parents) if a != root(a)]


def interior_moments(ratio, words):
    # Condition on the 18 boundary spins and then on the central spin. The
    # six other interior spins become independent and each has six neighbours.
    a, b = ratio.numerator, ratio.denominator
    total = [a**(6-k)*b**k + b**(6-k)*a**k for k in range(7)]
    signed = [a**(6-k)*b**k - b**(6-k)*a**k for k in range(7)]
    fields = [sum(1 << j for j, v in enumerate(BOUNDARY)
                  if sum(abs(x-y) for x, y in zip(v, leaf)) == 1) for leaf in DIRECTIONS]
    assert all(bin(field).count("1") == 5 for field in fields)
    values = np.empty((128, len(words)), dtype=np.int64)
    for col, word in enumerate(words):
        counts = [bin(int(word) & field).count("1") for field in fields]
        by_center = []
        for center in (0, 1):
            products = [1]
            for k in counts:
                pos, neg = total[k+center], signed[k+center]
                products = [n*pos for n in products] + [n*neg for n in products]
            by_center.append(products)
        plus, minus = by_center
        den = plus[0]+minus[0]
        for leaves, (p, m) in enumerate(zip(plus, minus)):
            for center_in_pattern, num in enumerate((p+m, p-m)):
                values[2*leaves+center_in_pattern, col] = (2*SCALE*num+den)//(2*den)
    return values


class BallMoments:
    def __init__(self, ratio, max_degree):
        self.words = boundary_representatives()
        self.interior = interior_moments(ratio, self.words)
        self.sign = np.fromiter(((-1)**bin(x).count("1") for x in range(1 << len(BOUNDARY))),
                                dtype=np.int8)
        pairs = translation_rows(max_degree)
        self.rows = np.asarray([self.averaged(a)-self.averaged(b) for a, b in pairs],
                               dtype=np.int64)
        self.g1 = self.averaged(mask([ORIGIN, (1, 0, 0)]))
        self.diagonal = self.averaged(mask([ORIGIN, (1, 1, 0)]))
        self.axial = self.averaged(mask([ORIGIN, (2, 0, 0)]))

    @lru_cache(None)
    def averaged(self, pattern):
        images = orbit(pattern)
        total = np.zeros(len(self.words), dtype=np.int64)
        for image in images:
            inner, outer = image % 128, image >> 7
            total += self.sign[self.words & outer] * self.interior[inner]
        return total * (len(GROUP)//len(images))

    def dual(self, target, divisor, cuts, tolerance=0.000006, relax=False):
        rows = self.rows / TOTAL_SCALE
        blocks = [rows, -rows]
        if cuts:
            blocks.append(-np.asarray([cut[0] for cut in cuts]) / TOTAL_SCALE)
        matrix = np.vstack(blocks)
        cost = -target/(divisor*TOTAL_SCALE)
        normalization = np.ones(len(cost))
        if relax:
            # A slack keeps the discovery problem feasible even when the physical
            # hypothesis is false. The verifier below only uses its multipliers.
            column = np.r_[np.zeros(2*len(rows)), -np.ones(len(cuts))]
            matrix = np.column_stack((matrix, column))
            cost = np.r_[cost, 1.0]
            normalization = np.r_[normalization, 0.0]
        result = linprog(cost, A_eq=[normalization],
                         b_eq=[1], A_ub=matrix, b_ub=np.full(len(matrix), tolerance),
                         bounds=(0, None), method="highs-ds")
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def certify(self, target, divisor, cuts, result):
        # Each cut is (integer row, rounding error, coefficient of m_+^2).
        marginals = result.ineqlin.marginals
        nrows = len(self.rows)
        proposed = marginals[nrows:2*nrows]-marginals[:nrows]
        denominator = 10**6
        coeff = [round(float(y)*denominator) for y in proposed]
        positive = [max(0, round(-float(y)*denominator)) for y in marginals[2*nrows:]]
        residual = np.asarray(target, dtype=object) * denominator
        for y, row in zip(coeff, self.rows):
            if y:
                residual -= (divisor*y) * row.astype(object)
        error = Fraction(1, 2*SCALE) + Fraction(sum(map(abs, coeff)), denominator*SCALE)
        mass_coefficient = Fraction()
        for y, (row, radius, zero_mode) in zip(positive, cuts):
            if y:
                residual += (divisor*y) * row.astype(object)
                error += Fraction(y, denominator)*radius
                mass_coefficient += Fraction(y, denominator)*zero_mode
        bound = Fraction(int(residual.max()), divisor*TOTAL_SCALE*denominator) + error
        return bound, mass_coefficient

    def walk_moments(self):
        moments, returns = [], []
        counts = Counter({ORIGIN: 1})
        for n in range(5):
            den = 6**n
            raw = np.zeros(len(self.words), dtype=np.int64)
            for delta, ways in counts.items():
                if delta == ORIGIN:
                    raw += ways * TOTAL_SCALE
                else:
                    a = next(v for v in SITES if add(v, delta) in INDEX)
                    raw += ways * self.averaged(mask([a, add(a, delta)]))
            moments.append((2*raw+den)//(2*den))
            returns.append(Fraction(counts[ORIGIN], den))
            following = Counter()
            for x, ways in counts.items():
                for step in DIRECTIONS:
                    following[add(x, step)] += ways
            counts = following
        return moments, returns


@lru_cache(None)
def scaled_returns(nmax):
    scale = 1 << 56
    counts = [scale, scale//6]
    previous, last, central, denominator = 1, 3, 2, 36
    for n in range(2, nmax+1):
        count, rem = divmod((10*n*n-10*n+3)*last-9*(n-1)**2*previous, n*n)
        assert rem == 0
        central = central*(4*n-2)//n
        denominator *= 36
        counts.append((scale*central*count)//denominator)
        previous, last = last, count
    return counts


def walk_resolvent(s, nmax=4000):
    # Enclose integral 1/(d-gamma), d=1+s/6. At s=0 use the elementary
    # Watson bounds. At positive mass use monotone p_{2n} for the tail.
    from verify_free_channel import improve_watson
    low_w, high_w = improve_watson()
    assert Fraction(1516349,10**6) < low_w < high_w < Fraction(1516425,10**6)
    if not s:
        return Fraction(1), Fraction(1516349,10**6), Fraction(1516425,10**6)
    d = 1+s/6
    t = 1/d**2
    scale = 1 << 56
    lower_power = scale*d.denominator//d.numerator
    upper_power = lower_power+1
    lo, hi = 0, 0
    for pn in scaled_returns(nmax):
        lo += lower_power*pn//scale
        hi += (upper_power*(pn+1)+scale-1)//scale
        lower_power = lower_power*t.numerator//t.denominator
        upper_power = (upper_power*t.numerator+t.denominator-1)//t.denominator
    tail = Fraction(upper_power*(pn+1), scale**2)/(1-t)
    return d, Fraction(lo, scale), Fraction(hi, scale)+tail


def spectral_cut(coeff, moments, returns, c0, resolvent):
    # P is a square or (1-gamma^2) times a square, normalized by its coefficient l1 norm.
    total = sum(map(abs, coeff))
    a = [Fraction(int(c), int(total)) for c in coeff]
    zero_mode = sum(a)
    d, low, high = resolvent
    at_d = sum(c*d**j for j,c in enumerate(a))
    correction = sum(c*sum(d**(j-1-i)*returns[i] for i in range(j))
                     for j,c in enumerate(a))
    integral = (at_d*(high if at_d >= 0 else low)-correction)/6
    assert integral >= 0 and zero_mode >= 0
    row_coeff = [-c for c in a] + [Fraction()]*(5-len(a))
    row_coeff[0] += c0*integral
    row_coeff[1] -= 2*integral
    den = lcm(*(c.denominator for c in row_coeff))
    sums = np.zeros(len(moments[0]), dtype=object)
    for c, row in zip(row_coeff, moments):
        if c:
            sums += int(c*den)*row.astype(object)
    rounded = ((2*sums+den)//(2*den)).astype(np.int64)
    moment_error = Fraction(1, 2*SCALE) + Fraction(1, 2*TOTAL_SCALE)
    error = sum(map(abs, row_coeff[1:]))*moment_error + Fraction(1, 2*TOTAL_SCALE)
    return rounded, error, zero_mode


def energy_test(ratio, system, target, divisor, s=Fraction(), relax=False, local_bounds=()):
    resolvent = walk_resolvent(s)
    d, low, high = resolvent
    u = (ratio-1)/(ratio+1)
    c0 = 1/u+u
    moments, returns = system.walk_moments()
    # Floating eigenvectors only select polynomials. Each chosen polynomial is
    # reconstructed as an exact square before its bound is used.
    coeff_column = float(c0)-2*moments[1]/TOTAL_SCALE
    deficit = [(float(d**j*(high+low)/2-sum(d**(j-i-1)*returns[i] for i in range(j)))/6)
               *coeff_column-row/TOTAL_SCALE
               for j,row in enumerate(moments)]
    cuts = list(local_bounds)
    seen = set()
    for iteration in range(10):
        result = system.dual(target, divisor, cuts, relax=relax)
        expected = [np.dot(row, result.x[:len(system.words)]) for row in deficit]
        added = False
        for localizer, size in ((False, 3), (True, 2)):
            matrix = [[expected[i+j]-(expected[i+j+2] if localizer else 0.0)
                       for i in range(size)] for j in range(size)]
            eig, vec = np.linalg.eigh(matrix)
            for k in np.where(eig < -0.00004)[0]:
                q = np.round(10000*vec[:, k]).astype(np.int64)
                coeff = np.convolve(q, q)
                if localizer:
                    coeff = np.convolve(coeff, (1, 0, -1))
                key = tuple(coeff.tolist())
                if key not in seen:
                    seen.add(key)
                    cuts.append(spectral_cut(key, moments, returns, c0, resolvent))
                    added = True
        if not added:
            break
    result = system.dual(target, divisor, cuts, relax=relax)
    return system.certify(target, divisor, cuts, result)


def complete_bipartite(ratio):
    total, nearest, distance_two = Fraction(), Fraction(), Fraction()
    for left, right in product(range(7), repeat=2):
        equal = left*right + (6-left)*(6-right)
        weight = comb(6, left)*comb(6, right)*ratio**equal
        total += weight
        nearest += weight*Fraction((2*left-6)*(2*right-6), 36)
        distance_two += weight*Fraction((2*right-6)**2-6, 30)
    g = nearest/total
    variance = (343+612*g+1445*distance_two/total)/2400
    return g, variance


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("ratio", nargs="*", type=Fraction)
    args = parser.parse_args()
    for ratio in args.ratio or [Fraction(25,16), Fraction(19,12)]:
        system = BallMoments(ratio, 4)
        assert len(system.words) == 3236 and len(system.rows) == 102
        vnum = 343*TOTAL_SCALE + 612*system.g1 + 1156*system.diagonal + 289*system.axial
        if ratio == Fraction(25,16):
            bp_g, bp_v = complete_bipartite(ratio)
            for t, den, lower in ((system.g1, 1, bp_g), (vnum, 2400, bp_v)):
                upper, mass = system.certify(t, den, [], system.dual(t, den, []))
                assert mass == 0 and lower <= upper < lower+Fraction(1, 10**5)
                print("local-only maximum", float(lower), float(upper), flush=True)
        for label, t, den in (("g1", system.g1, 1), ("V", vnum, 2400)):
            upper, mass = energy_test(ratio, system, t, den)
            print(ratio, label, "upper plus coefficient times m^2:",
                  upper, mass, float(upper), float(mass), flush=True)


if __name__ == "__main__":
    main()
