"""Directed integer enclosures for the 14 exterior-source comparisons.

Only the proposed cap uses floating point. All acceptance tests round down.
The baseline cap used in the original verifier is checked from below on a
non-corner field, so the selected corner cap also passes its original cutoff.
"""
from fractions import Fraction
from itertools import product
from math import ceil

import numpy as np


HALF_BITS = 25
MASK = (1 << HALF_BITS) - 1
UNIT = 1 << (2 * HALF_BITS)
AMPLITUDE = 1 << 52


def coefficients(ratio):
    lower, remainder = divmod(UNIT * ratio.numerator, ratio.denominator)
    assert 0 <= lower <= lower + bool(remainder) <= UNIT
    return lower, lower + bool(remainder)


def multiply_lower(values, lower, upper):
    """Floor a signed product using outward dyadic factors and 25-bit limbs."""
    assert max(abs(int(values.min())), abs(int(values.max()))) <= 2**54
    multiplier = np.where(values < 0, upper, lower)
    hi, lo = values >> HALF_BITS, values & MASK
    top, tail = multiplier >> HALF_BITS, multiplier & MASK
    carry = hi * tail + lo * top + ((lo * tail) >> HALF_BITS)
    return hi * top + (carry >> HALF_BITS)


def average_lower(values, ratios):
    """Componentwise lower enclosure under the normalized transfer factors."""
    out = values.copy()
    assert max(abs(int(out.min())), abs(int(out.max()))) < 2**53
    for j, ratio in enumerate(ratios):
        lo, hi = coefficients(ratio / (1 + ratio))
        width = 1 << j
        pairs = out.reshape(-1, 2 * width)
        left, right = pairs[:, :width], pairs[:, width:]
        delta = right - left
        left += multiply_lower(delta, lo, hi)
        right += multiply_lower(-delta, lo, hi)
    return out


def average_float(values, ratios):
    out = values.copy()
    for j, ratio in enumerate(ratios):
        width = 1 << j
        pairs = out.reshape(-1, 2 * width)
        left, right = pairs[:, :width].copy(), pairs[:, width:].copy()
        pairs[:, :width] = left + float(ratio) * right
        pairs[:, width:] = right + float(ratio) * left
    return out


def normalize_lower(entries, owners):
    peak = max(map(abs, entries))
    assert peak > 0
    bounded = np.fromiter((int(v * AMPLITUDE // peak) for v in entries),
                          dtype=np.int64, count=len(entries))
    return bounded[owners]


def check_products():
    probes = np.array([-2**54, -AMPLITUDE + 1, -13, -1, 0,
                       1, 27, AMPLITUDE - 5, 2**54], dtype=np.int64)
    for factor in (Fraction(0), Fraction(1, 3), Fraction(441899, 912739),
                   Fraction(UNIT - 1, UNIT), Fraction(1)):
        lo, hi = coefficients(factor)
        got = multiply_lower(probes, lo, hi)
        expected = [int(v) * (hi if v < 0 else lo) // UNIT for v in probes]
        assert got.tolist() == expected
        assert all(x <= int(v) * factor for x, v in zip(expected, probes))
    raw = np.array([3, -841, 910, -12, -100871, 403, 141293, -453], dtype=np.int64)
    factors = [Fraction(31, 50), Fraction(47, 59), Fraction(5, 7)]
    exact = list(map(Fraction, map(int, raw)))
    for j, t in enumerate(factors):
        exact = [(v + t * exact[i ^ (1 << j)]) / (1 + t) for i, v in enumerate(exact)]
    lower = average_lower(raw, factors)
    assert all(int(lo) <= value <= int(lo) + 4 for lo, value in zip(lower, exact))


class CornerCheck:
    def __init__(self, base):
        self.base = base
        check_products()

    def coarse_bound(self, values, weights, words, owners, scale, ratios):
        # Every unrestricted exterior-field value is a weighted average.
        return Fraction(max(map(int, values)), int(scale)), None

    def all_corners(self, r, values, weights, owners, scale, ratios, coarse, saved=None):
        base = self.base
        corners = list(product((-1, 1), repeat=3))
        touched = [[i for i, p in enumerate(base.geometry.BOUNDARY)
                    if p in base.fields.neighbours(c)] for c in corners]
        assert all(len(sites) == 3 for sites in touched)
        degree = [sum(i in sites for sites in touched) for i in range(len(ratios))]
        assert degree == [0 if max(map(abs, p)) == 2 else 2 for p in base.geometry.BOUNDARY]
        reduced = [v * r**d for v, d in zip(ratios, degree)]
        assert all(0 < v <= 1 for v in reduced)
        assert len(ratios) == 18 and len(owners) == 2**18
        words = np.arange(len(owners), dtype=np.int32)
        pop = np.fromiter((bin(i).count("1") for i in range(len(words))), dtype=np.int8)
        counts = [pop[words & sum(1 << j for j in sites)] for sites in touched]
        classes = base.fields.corner_classes(corners)
        assert len(classes) == 14
        a, b = r.numerator, r.denominator
        trace = [a**(6-j) * b**j + a**j * b**(6-j) for j in range(4)]
        ratios_table = [Fraction(x, trace[0]) for x in trace]

        def indices(source):
            return (3 - count if source & (1 << j) else count for j, count in enumerate(counts))

        if saved is None:
            peak = max(weights)
            norm = np.array([w / peak for w in weights])[owners]
            floats = np.array([float(q) for q in ratios_table])
            signed = np.array([int(v) / scale for v in values])[owners]
            greatest = -np.inf
            for source in classes:
                value = norm.copy()
                for index in indices(source):
                    value *= floats[index]
                quotients = average_float(value * signed, reduced) / average_float(value, reduced)
                assert np.isfinite(quotients).all()
                greatest = max(greatest, float(quotients.max()))
            saved = Fraction(ceil(greatest * 10**9) + 5, 10**9)
        assert saved <= coarse

        # Positive normalization of each tensor factor preserves this sign test.
        residual = [int(w) * (saved.numerator * scale - saved.denominator * int(v))
                    for v, w in zip(values, weights)]
        start = normalize_lower(residual, owners)
        opposite = normalize_lower([-x for x in residual], owners)
        assert average_lower(opposite, ratios).max() >= 0, "check unrestricted cutoff directly"
        low, high = (np.array(seq, dtype=np.int64) for seq in zip(*map(coefficients, ratios_table)))
        for k, source in enumerate(classes, 1):
            field = start.copy()
            for index in indices(source):
                field = multiply_lower(field, low[index], high[index])
            enclosure = average_lower(field, reduced)
            assert enclosure.min() >= 0, (r, source, int(enclosure.min()))
            if k in (1, 7, 14):
                print(f"directed integer enclosures: {k}/14 corner classes", flush=True)
        return saved
