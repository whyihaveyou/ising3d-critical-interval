"""Propagate a checked ordered endpoint to a closed, explicit left interval."""

from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import argparse
import json
from math import floor
from pathlib import Path

import STAGE23_HOLDBACK_GREEN as green
from STAGE29_ORDERED_corner import class_graph, odds

HERE = Path(__file__).resolve().parent
GRID = 10**10
SOURCE = HERE / "STAGE21_ORDERED_780_493.json"
KNOWN_SOURCE_SHA = "b8c8cec1644bf8bd29978d514643a7044377eb12fcd0e157d950cd2980bd2cb6"


class Interval:
    """Exact closed rational intervals for the positive local factors."""
    def __init__(self, low, high=None):
        self.lo = F(low)
        self.hi = self.lo if high is None else F(high)
        assert self.lo <= self.hi

    def __add__(self, other):
        other = interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __sub__(self, other):
        other = interval(other)
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __rsub__(self, other):
        return interval(other) - self

    def __mul__(self, other):
        other = interval(other)
        ends = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(ends), max(ends))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = interval(other)
        assert other.lo > 0
        return self * Interval(1/other.hi, 1/other.lo)

    def __rtruediv__(self, other):
        return interval(other) / self

    def __pow__(self, n):
        assert n >= 0 and self.lo >= 0
        return Interval(self.lo**n, self.hi**n)


def interval(x):
    return x if isinstance(x, Interval) else Interval(x)


def local_factors(s):
    # The four corner-source odds followed by the axis and face flip odds.
    def bond(x):
        return (s+x)/(1+s*x)
    v = s**3
    leaf = bond(s**5)
    outer = bond(s**4*bond(leaf**5))**3
    match = bond(s**4*bond(leaf**2/v))
    mismatch = bond(s**4*bond(leaf**2*v))
    result = []
    for n in range(4):
        inner = match**n*mismatch**(3-n)
        result.append((outer-v)*(inner+v)/((1-v*outer)*(1+v*inner)))
    theta = (s**2+s**4)/(1+s**6)
    axial = s
    for _ in range(3):
        axial = bond(s**4*axial)
    for rho in (theta**2*axial, theta):
        result.append(rho*(rho/theta-rho)/(1-rho*rho/theta))
    return result


def shared_edges():
    boundary = sorted(p for p in product(range(-2, 3), repeat=3) if sum(map(abs, p)) == 2)
    edges = set()
    for p in product(range(-2, 3), repeat=3):
        if sorted(map(abs, p)) != [0, 1, 2]:
            continue
        neighbours = [i for i, b in enumerate(boundary) if sum(abs(x-y) for x,y in zip(p,b)) == 1]
        assert len(neighbours) == 2
        i,j = sorted(neighbours, key=lambda i: -max(map(abs, boundary[i])))
        assert (max(map(abs, boundary[i])), max(map(abs, boundary[j]))) == (2, 1)
        edges.add((i, j))
    assert len(edges) == 24
    return edges


def combine(archive, safe, sizes, ordered_pairs):
    edges = shared_edges()
    caps = {}
    for key, rec in archive["class_bounds"].items():
        high = {word for word, _, _ in rec["words"]}
        assert len(high) == len(rec["words"]) and len(high) < 2**18
        axis_targets, face_targets = Counter(), Counter()
        for f, i, j in rec["words"]:
            assert 0 <= f < 2**18 and (i,j) in edges and ((f >> i) ^ (f >> j)) & 1
            axis, face = f^(1 << i), f^(1 << j)
            assert axis not in high and face not in high
            axis_targets[axis] += 1
            face_targets[face] += 1
        C, D = F(rec["upper"]), F(rec["low"])
        assert D <= C
        if high:
            B = max(axis_targets[f]/safe[4] + face_targets[f]/safe[5]
                    for f in axis_targets.keys() | face_targets.keys())
            caps[int(key)] = (B*C+D)/(1+B)
        else:
            assert C == D
            caps[int(key)] = C
    assert caps.keys() == sizes.keys()
    shifts = dict.fromkeys(sizes, F(0))
    seen = set()
    for i,j,v in archive["class_dual"]:
        assert v > 0 and (i,j) in ordered_pairs and (i,j) not in seen
        seen.add((i,j))
        y = F(v, archive["class_denominator"])
        shifts[i] -= y * safe[ordered_pairs[i,j]] / sizes[i]
        shifts[j] += y / sizes[j]
    return max(caps[k] + shifts[k] for k in caps) + F(archive["radius"])


def slope_bound(data):
    d = data["denominator"] * 10**6
    trans, standard, deleted = [F(sum(abs(row[-1]) for row in data["rows"][key]), d)
                                for key in ("translations", "ordinary", "walsh")]
    assert (trans < 4 and standard < 21 and deleted < F(1, 1000))
    constants = green.analytic_constants()
    for tag, coeffs in green.INPUTS.values():
        poly = green.symmetric_polynomial(tag, coeffs)
        lo, hi = green.integral_bounds(poly, constants)
        assert 0 < lo < hi < 1
    # Spin moments and their log-r derivatives are bounded by 1 and 72.
    # The four normalized reflection vectors each have dimension 112.
    value = 2*trans + 112*(1+6)*standard + 64*deleted
    rate = 144*trans + 112*((1+6)*72+12)*standard + 32*(2*72+300)*deleted
    log_weight = 36 + 24*6 + 8*6 + 6*(2*6+16) + 12*6
    assert value < 17000 and rate < 1300000 and log_weight == 468
    assert rate + 2*log_weight*value < 18000000
    return 18000000


def verify(point, delta):
    assert __debug__ and F(3,2) < point - delta < point < F(8,5)
    raw = (HERE / f"STAGE30_UPPER_REVIEW_{point.numerator}_{point.denominator}.json").read_bytes()
    archive = json.loads(raw)
    source = SOURCE.read_bytes()
    assert F(archive["r"]) == point
    assert sha256(source).hexdigest() == KNOWN_SOURCE_SHA == archive["source_sha256"]
    rows = json.loads(source)["rows"]
    assert [len(rows[k]) for k in ("translations", "ordinary", "walsh")] == [104,9,7]
    assert {k for k, _ in rows["ordinary"]} == set(green.INPUTS) | {42, 45, 76, 83}
    factors = local_factors(1/point)
    assert factors[:4] == odds(point)
    safe = [F(floor(x*GRID), GRID) for x in factors]
    enclosure = local_factors(Interval(1/point, 1/(point-delta)))
    assert all(0 < cut < bounds.lo <= x <= bounds.hi for cut,bounds,x in zip(safe,enclosure,factors))
    sizes, pairs = class_graph()
    upper = combine(archive, safe, sizes, pairs)
    Q = F(archive["Q"])
    lower = -upper/Q
    assert upper == F(archive["U"]) < 0 < Q
    assert lower == F(archive["L"]) and floor(10**15*lower) == archive["floor_1e15_L"]
    # log(point/r) <= (point-r)/r throughout the test interval.
    K = slope_bound(json.loads(source))
    uniform_upper = upper + K*delta/(point-delta)
    assert uniform_upper < 0
    uniform_lower = -uniform_upper/Q
    assert F(floor(10**15*uniform_lower), 10**15) < uniform_lower
    print("endpoint", point, "delta", delta, "left", point-delta)
    print("frozen odds (units 1e-10)", [int(x*GRID) for x in safe])
    print("archive sha256", sha256(raw).hexdigest())
    print("floor(1e15 original squared-magnetization lower bound)", floor(lower*10**15))
    print("floor(1e15 uniform squared-magnetization lower bound)", floor(uniform_lower*10**15))
    print("PASS 24 bridge edges, 40 class odds, frozen high sets, 120 physical rows and uniform left interval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=F, default=F(5441,3439))
    parser.add_argument("--delta", type=F, default=F(1,10**14))
    args = parser.parse_args()
    verify(args.point, args.delta)

