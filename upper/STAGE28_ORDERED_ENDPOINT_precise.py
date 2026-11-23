"""Certify a weaker ordered endpoint with high-precision local spin moments.

The Stage 21 signed physical multipliers and corner-class shifts are fixed.
Only the coupling, integer accuracy and five Green integral inputs change.
Run ``directed`` once to store the rational caps, then ``raw`` to verify them.
"""

import argparse
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import hashlib
import json
from math import comb, floor, prod
from pathlib import Path

import numpy as np

import STAGE21_ORDERED as base
import STAGE21_ORDERED_active_rows as active
import STAGE23_HOLDBACK_GREEN as green_formula

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "STAGE21_ORDERED_780_493.json"
OLD_POINT = F(4449, 2812)
SPIN_UNIT = 1 << 38
W = base.stage.W
G = W.geometry
TOTAL = len(G.GROUP) * SPIN_UNIT
MULTIPLE = TOTAL // W.T
assert TOTAL == MULTIPLE * W.T and MULTIPLE == SPIN_UNIT // W.fixed.SCALE


class ConditionalMoments:
    """Sum every spin word on the seven inner sites before rounding once."""

    def __init__(self, model):
        self.words = model.ball.words
        self.sign = model.ball.sign
        boundary = (self.words.astype(np.int64) << 7)[None, :]
        words = boundary | np.arange(128, dtype=np.int64)[:, None]
        edges = [(x, y) for x in range(7) for y in range(x+1, len(G.SITES))
                 if sum(abs(u-v) for u, v in zip(G.SITES[x], G.SITES[y])) == 1]
        assert len(edges) == 36
        mismatches = np.zeros(words.shape, dtype=np.int16)
        for x, y in edges:
            mismatches += ((words >> x) ^ (words >> y)) & 1
        a, b = model.ratio.as_integer_ratio()
        weights = np.array([a**(36-j)*b**j for j in range(37)], dtype=object)
        table = weights[mismatches]
        for k in range(7):
            width = 1 << k
            for start in range(0, 128, 2*width):
                left = table[start:start+width].copy()
                right = table[start+width:start+2*width].copy()
                table[start:start+width] = left + right
                table[start+width:start+2*width] = left - right
        self.denominators = table[0].copy()
        assert np.array_equal(self.denominators, model.deleted.z)
        self.interior = ((2*SPIN_UNIT*table + self.denominators)
                         // (2*self.denominators)).astype(np.int64)
        assert np.all(abs(self.interior) <= SPIN_UNIT)
        # Compare both independent enumeration and normalization with the
        # older six-arm conditional spin contraction in all 414,208 entries.
        discrepancy = self.interior - MULTIPLE*model.ball.interior
        assert 2*int(abs(discrepancy).max()) <= MULTIPLE + 1

    @lru_cache(None)
    def averaged(self, pattern):
        return G.BallMoments.averaged(self, pattern)


def angular_moments(moments, phase):
    """Cosine monomials, as an integer sum over signed axial steps."""
    @lru_cache(None)
    def displacement(steps):
        if not any(steps):
            return TOTAL
        first = next(p for p in G.SITES if G.add(p, steps) in G.INDEX)
        return moments.averaged(G.mask((first, G.add(first, steps))))

    rows = []
    for exponents in phase.KEYS:
        paths = [[(n-2*j, comb(n, j)) for j in range(n+1)] for n in exponents]
        denom = 2**sum(exponents)
        assert 2*denom*TOTAL+denom < (1 << 63)
        acc = np.zeros(len(moments.words), dtype=np.int64)
        for steps in product(*paths):
            offsets, ways = zip(*steps)
            acc += prod(ways)*displacement(offsets)
        rows.append((2*acc+denom)//(2*denom))
    return rows


class AnalyticGreen:
    """Use upper endpoints for J(P) in the normalization of spectral_cut."""
    def __init__(self, phase):
        green_formula.check_harmonic()
        constants = green_formula.analytic_constants()
        self.bounds = {}
        for tag, vector in green_formula.INPUTS.values():
            poly = green_formula.symmetric_polynomial(tag, vector)
            coeff = tuple(phase.polynomial(tag, vector))
            assert coeff == tuple(poly.get(green_formula.canonical(k), 0) for k in phase.KEYS)
            self.bounds[coeff] = green_formula.integral_bounds(poly, constants)
        assert len(self.bounds) == 5

    def upper(self, poly):
        return 6*self.bounds[tuple(poly)][1]


def build_residual(model, source):
    record, denominator = source['rows'], source['denominator']
    moments = ConditionalMoments(model)
    phase = W.fixed.phase
    angular = angular_moments(moments, phase)
    gram = phase.reflection_features(moments)
    analytic = AnalyticGreen(phase)
    legacy = phase.AngularGreen(F(0))
    u = (model.ratio - 1)/(model.ratio + 1)
    bare = u + 1/u
    tests = active.active_tests(record['ordinary'])
    assert set(tests) == {29, 32, 33, 35, 42, 45, 69, 76, 83}
    assert {j: t for j, t in tests.items() if t[0] != 'extension'} == green_formula.INPUTS
    values = np.zeros(len(model.words), dtype=object)
    error, mass = F(0), F(0)
    translate = G.translation_rows(8)
    for index, weight in record['translations']:
        first, second = translate[index]
        row = moments.averaged(first) - moments.averaged(second)
        values -= weight*row.astype(object)
        error += F(abs(weight), SPIN_UNIT)
    for index, weight in record['ordinary']:
        tag, root = tests[index]
        if tag == 'extension':
            row, eps, slope = W.fixed.exact_reflection(root, bare, gram)
            prior_row = row
        else:
            poly = phase.polynomial(tag, root)
            row, eps, slope = phase.spectral_cut(poly, angular, bare, analytic)
            prior_row, _, _ = phase.spectral_cut(poly, angular, bare, legacy)
            assert 0 < analytic.upper(poly) <= legacy.upper(poly)
        _, coarse, old_error, old_slope = model.terms[index]
        assert slope == old_slope
        gap = max(abs(int(a)-MULTIPLE*int(b)) for a, b in zip(prior_row, coarse))
        assert F(gap, TOTAL) <= old_error*(1+F(1, MULTIPLE))
        values += weight*row.astype(object)
        # Both summands in the production radius are inverse-grid terms.
        # The functions above use legacy module constants only for this radius.
        error += weight*eps/MULTIPLE
        mass += weight*slope
    for a, toggle, weight in record['walsh']:
        values += weight*MULTIPLE*model.deleted.row(a, toggle).astype(object)
        error += weight*W.fixed.local.GRID
    weight_scale = denominator*W.DEN
    return values, weight_scale*TOTAL, error/weight_scale, mass/weight_scale


def run(point, mode):
    assert __debug__ and F(791, 500) < point < OLD_POINT
    archive = HERE / f'STAGE28_ORDERED_ENDPOINT_precise_{point.numerator}_{point.denominator}.json'
    source = json.loads(SOURCE.read_text())
    base.POINT = point
    d, sizes, edges = base.class_geometry()
    model, pool, layout = base.make_model()
    base.validate_rows(source['rows'], pool)
    offsets = base.class_offsets(source['class_dual'], sizes, edges)
    residual, scale, radius, mass = build_residual(model, source)
    assert mass == base.stage.MASS > 0 and d > base.star.LOW and radius > 0
    if mode == 'raw' or (mode == 'directed' and archive.exists()):
        saved = json.loads(archive.read_text())
        caps = {int(k):(F(v), w) for k, (v, w) in saved['caps'].items()}
    else:
        caps = base.propose_caps(model, residual, scale, layout)
    assert set(caps) == set(sizes)
    owner = max(caps, key=lambda c: caps[c][0] + offsets[c])
    cap, field = caps[owner]
    weights = base.stage.corner_weights(model, owner, layout[1], layout[3])
    weights = base.stage.direct_field(weights, layout[2], layout[4], layout[5], field)
    witness = F(int(weights @ residual[model.owners]), int(weights.sum())*scale)
    assert 0 <= cap-witness < F(1, 10**7)
    upper = cap + offsets[owner] + radius
    lower = -upper/mass
    print(f'r={point} active=({owner},{field}) Q={mass} '
          f'1e15 floors: radius={floor(10**15*radius)}, U={floor(10**15*upper)}, '
          f'L={floor(10**15*lower)}, d-d0={floor(10**15*(d-base.star.LOW))}', flush=True)
    if mode == 'probe':
        return
    assert upper < 0 < F(floor(10**15*lower), 10**15) < lower
    bound = {k: z for k, (z, _) in caps.items()}
    base.fields.check_fields(model, residual, scale, layout, bound, raw=mode == 'raw')
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    certificate = dict(r=str(point), d=str(d), source=SOURCE.name, source_sha256=digest,
                       spin_unit=SPIN_UNIT, Q=str(mass), radius=str(radius), U=str(upper),
                       L=str(lower), active=[owner,field], floor_1e15_L=floor(10**15*lower),
                       caps={str(k):[str(z), w] for k,(z,w) in caps.items()})
    if mode == 'directed' and not archive.exists():
        archive.write_text(json.dumps(certificate, indent=2)+'\n')
    assert json.loads(archive.read_text()) == certificate
    print(f'PASS {mode}, {len(layout[0])*len(model.owners)} fields, '
          f'floor(1e15 m2 bound)={floor(10**15*lower)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ratio', type=F)
    parser.add_argument('mode', choices=('probe','directed','raw'))
    args = parser.parse_args()
    run(args.ratio, args.mode)
