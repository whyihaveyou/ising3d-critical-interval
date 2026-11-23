"""Rebuild the stage-11 residual and test its sign on an explicit exterior field.

Run with --all-corners to verify the archived upper envelope for every source
and exterior word. --domain-ladder exercises the physical-row construction
at the other breakpoints and its historical lower-coupling guard.
"""
import argparse
from fractions import Fraction as F
from itertools import product
import json
from math import floor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent / "physical"
sys.path.insert(0, str(ROOT))
import verify_deleted_walsh_order as walsh

base = walsh.base
RATIO = F(31, 20)
KEY = "order 31/20 bridges expanded corners"


def test_message_factors():
    # Test the actual fixed-point word widths and the star trace used by the
    # 31-bit diagonal-message kernel at each indicated rational input.
    for r in (F(77, 50), F(309, 200), RATIO, F(311, 200), F(7789, 5000)):
        a, b = r.numerator, r.denominator
        for degree in range(4):
            c = [a**j*b**(degree-j) for j in range(degree+1)]
            max_sum = max(left+right for left, right in zip(c, reversed(c)))
            shift = max_sum.bit_length()-1
            raw = (2**31-1) * max_sum
            assert max(c) < 2**64 and raw < 2**128 and raw >> shift < 2**32
        for degree, mixed in ((2, 2*r/(r*r+1)), (3, r*(r+1)/(r**3+1))):
            aligned = a**degree + b**degree
            for j in range(1, degree):
                partial = a**j*b**(degree-j) + a**(degree-j)*b**j
                assert F(partial, aligned) == mixed


def check_table(ratio, ball, terms, table):
    # Free two-point lower bounds occupy the final seven columns of rebuild.
    assert len(table) == 7
    for (label, floor), (_, row, error, slope) in zip(table.items(), terms[-7:]):
        shift = tuple(map(int, label))
        p = next(p for p in walsh.geometry.SITES
                 if walsh.geometry.add(p, shift) in walsh.geometry.INDEX)
        pair = walsh.geometry.mask([p, walsh.geometry.add(p, shift)])
        raw = ball.averaged(pair)
        expected = raw - walsh.T * floor // 10**10
        assert np.array_equal(row, expected), (ratio, label)
        assert error == walsh.fixed.local.ROUND and slope == 0


def test_ladder(green):
    for ratio in (F(77, 50), F(309, 200)):
        try:
            base.rebuild(ratio, green, expanded=True)
        except ValueError as err:
            assert "expanded physical inputs" in str(err)
        else:
            raise AssertionError(f"baseline guard changed at {ratio}")
    for ratio, table in ((F(311, 200), base.FREE_31_20),
                         (F(39, 25), base.FREE_39),
                         (F(25, 16), base.FREE_TARGET)):
        ball, _, terms = base.rebuild(ratio, green, expanded=True)
        check_table(ratio, ball, terms, table)


def build_residual(model, record):
    # The values array contains the signed orbit residual, in units of DEN*T.
    values = np.zeros(len(model.words), dtype=object)
    radius = F(0)
    Q = F(0)
    for index, weight in record["translations"]:
        radius += F(abs(weight), walsh.DEN * walsh.fixed.SCALE)
        values -= weight * model.ball.rows[index].astype(object)
    for index, weight in record["ordinary"]:
        assert weight > 0
        _, row, error, slope = model.terms[index]
        values += weight * row.astype(object)
        radius += F(weight, walsh.DEN) * error
        Q += F(weight, walsh.DEN) * slope
    for a, toggle, weight in record["walsh"]:
        assert weight > 0
        values += weight * model.deleted.row(a, toggle).astype(object)
        radius += F(weight, walsh.DEN) * walsh.fixed.local.GRID
    return values, radius, Q


def corner_geometry(model):
    corners = list(product((-1, 1), repeat=3))
    boundary = walsh.geometry.BOUNDARY
    adjacent = [[j for j, site in enumerate(boundary)
                 if sum(abs(x-y) for x, y in zip(site, corner)) == 1]
                for corner in corners]
    assert len(adjacent) == 8 and all(len(group) == 3 for group in adjacent)
    degrees = [sum(j in group for group in adjacent) for j in range(len(boundary))]
    ratios = [model.ratio**k * s for k, s in zip(degrees, model.ratios)]
    assert all(0 < s <= 1 for s in ratios)
    words = np.arange(len(model.owners), dtype=np.int32)
    popcount = np.fromiter((bin(int(w)).count("1") for w in words), dtype=np.int8)
    counts = [popcount[words & sum(1 << j for j in group)] for group in adjacent]
    a, b = model.ratio.numerator, model.ratio.denominator
    trace = np.array([a**(6-k)*b**k + b**(6-k)*a**k for k in range(4)],
                     dtype=object)
    return walsh.fields.corner_classes(corners), counts, ratios, trace, words, popcount


def corner_weights(model, source, counts, traces):
    weights = model.weights[model.owners].copy()
    for bit, tally in enumerate(counts):
        weights *= traces[3-tally if (source >> bit) & 1 else tally]
    return weights


def direct_field(weights, ratios, words, popcount, field):
    # Build the tensor-product kernel for one field by Hamming counts,
    # independently of the production butterfly transform.
    product_weights = weights.copy()
    diff = words ^ field
    for ratio in sorted(set(ratios)):
        mask = sum(1 << i for i, s in enumerate(ratios) if s == ratio)
        count = bin(mask).count("1")
        kernel = np.array([ratio.numerator**j * ratio.denominator**(count-j)
                           for j in range(count+1)], dtype=object)
        product_weights *= kernel[popcount[diff & mask]]
    return product_weights


def check_sign(all_corners, domain_ladder):
    test_message_factors()
    green = walsh.fixed.phase.AngularGreen(F(0))
    model = walsh.Certificate(RATIO, "order", green, False, bridges=True,
                              expanded=True, corners=False)
    check_table(RATIO, model.ball, model.terms, base.FREE_31_20)
    archive = json.loads(walsh.FILE.read_text())
    record = archive[KEY]
    assert (len(record["translations"]), len(record["ordinary"]),
            len(record["walsh"])) == (91, 24, 226)
    values, radius, Q = build_residual(model, record)
    assert Q == F(record["mass"]) > 0 and radius >= 0
    lower = F(record["value"])
    U = -Q * lower
    cap = U - radius
    scale = walsh.DEN * walsh.T
    raw = values[model.owners]
    classes, counts, ratios, trace, words, popcount = corner_geometry(model)

    # The all-zero corner source and all-zero shell field already obstruct
    # every negative residual cap. Sum this field directly over the spins.
    source, field = 0, 0
    assert source in classes
    weights = corner_weights(model, source, counts, trace)
    direct = direct_field(weights, ratios, words, popcount, field)
    numerator = int(direct @ raw)
    denominator = int(np.sum(direct))
    assert denominator > 0
    x = F(numerator, denominator*scale)
    assert numerator == walsh.fields.transform(weights * raw, ratios)[field]
    assert cap == F(15423467, 250000000) and 0 < x <= cap
    assert floor(10**12*x) == 61320152447
    assert floor(10**12*(x+radius)) == 61320504808
    assert U >= x + radius > 0 and lower == -U/Q < 0
    assert F(archive["order 894/565"]["value"]) > 0

    print("Q =", Q, flush=True)
    print("rounding radius =", radius)
    print("certified corner cap =", cap)
    print("U = cap + radius =", U)
    print("order lower -U/Q =", lower)
    print("witness source and field =", source, field)
    print("floor(10^12 * raw field residual) =", floor(10**12*x))
    print("floor(10^12 * (raw residual + radius)) =", floor(10**12*(x+radius)))
    if all_corners:
        # This checks every candidate against cap as an integer sign, with
        # the physical rows and sign built above. No floating optimizer runs.
        bound_delta = cap.numerator * scale - cap.denominator * raw
        for n, source in enumerate(classes, 1):
            weights = corner_weights(model, source, counts, trace)
            checked = walsh.fields.transform(weights * bound_delta, ratios)
            assert np.min(checked) >= 0, source
            if n in (1, 7, 14):
                print(f"all-field integer caps checked for {n}/{len(classes)} classes",
                      flush=True)
        assert len(classes)*len(words) == 3670016
    if domain_ladder:
        test_ladder(green)
    print("PASS: the stage-11 order lower bound is negative", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-corners", action="store_true")
    parser.add_argument("--domain-ladder", action="store_true")
    options = parser.parse_args()
    check_sign(options.all_corners, options.domain_ladder)
