"""Rebuild an ordered-phase certificate at the indicated cubic-lattice coupling.

All final field transforms, row radii, and zero-mode residuals use integers
or rational arithmetic. The zero-field angular Green bound is coupling-independent.
"""
import argparse
import json
import os
from fractions import Fraction as F
from itertools import product
from math import ceil, prod
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import qr

import verify_conditional_energy_star as geometry
import verify_ordered_coupling_extension as fixed
import verify_exterior_field_energy as fields


PROOF_FILE = Path(__file__).with_name("ordered_exterior_field_multipliers.json")
FREE_RATIO = F(79, 50)
# Lower endpoints, in units of 10^-10, for the free diagonal-section calculation.
FREE_79 = {"100":3388925464, "110":2135546762, "111":1634892710,
           "210":1286597457, "211":1127311533, "220":938940336}
# Independent lower endpoints from the same finite free subgraph at
# r=39/25. The 200 displacement uses the GKS product bound and is weaker
# than a direct finite-subgraph input.
FREE_39 = {"100":3141653635, "110":1864539814, "111":1371028141,
           "210":1043882593, "211":887400778, "220":718697603,
           "200":986998756}
# Independently replayed outward lower endpoints from the same finite free
# subgraph at r=31/20.  The earlier rejection came from a direct audit that
# omitted the reflected rightmost layer.  The corrected full-layer transfer
# audit and a fresh 486-section message rerun now support these six floors.
# The 200 displacement is supplied by the GKS product bound from 100.
FREE_31_20 = {"100":3032219040, "110":1747852657, "111":1259486998,
              "210":943773019, "211":789985677, "220":630913305,
              "200":919435230}
FREE_TARGET = {"100":3170321192, "110":1895460819, "111":1400810325,
               "210":1070877973, "211":913839564, "220":742699851,
               "200":1318704742}


def trace_and_field_ratios(r, words, ray_length):
    a, b = r.numerator, r.denominator
    masks = [
        sum(1 << j for j, site in enumerate(geometry.BOUNDARY)
            if site in fields.neighbours(arm))
        for arm in geometry.DIRECTIONS
    ]
    assert len(masks) == 6 and all(bin(mask).count('1') == 5 for mask in masks)
    spin_sum = [a**(6-k)*b**k + b**(6-k)*a**k for k in range(7)]
    traces = []
    for word in words:
        minus = [bin(mask & word).count('1') for mask in masks]
        traces.append(sum(prod(spin_sum[c+j] for j in minus) for c in (0, 1)))
    edges = [(geometry.INDEX[x], geometry.INDEX[y])
             for x in geometry.DIRECTIONS for y in fields.neighbours(x)]
    assert len(edges) == 36
    for i in (0, 7, len(words)//2, len(words)-1):
        z = 0
        for interior in range(128):
            config = words[i] << 7 | interior
            n = sum((config >> x ^ config >> y) & 1 for x, y in edges)
            z += a**(36-n) * b**n
        assert traces[i] == z
    s = 1/r
    message = s
    for _ in range(ray_length):
        message = (s + s**4 * message)/(1 + s**5 * message)
    ratios = []
    inner = set(geometry.INTERIOR)
    for x in geometry.BOUNDARY:
        neighbours = fields.neighbours(x)
        assert not neighbours.intersection(geometry.BOUNDARY)
        degree = len(neighbours & inner)
        assert degree in (1, 2)
        if degree == 1:
            direction = tuple(v//2 for v in x)
            assert direction in geometry.DIRECTIONS
            prev = x
            for j in range(ray_length):
                site = tuple((j+3)*v for v in direction)
                nxt = geometry.add(site, direction)
                sides = fields.neighbours(site) - {prev, nxt}
                assert len(sides) == 4 and all(sum(t != 0 for t in y) == 2 for y in sides)
                prev = site
            ratios.append(s**4 * message)
        else:
            ratios.append(s**4)
    return np.array(traces, dtype=object), ratios


def knight_fields(ratio, words, z, original_ratios):
    """Trace all 24 sites of type (2,1,0); the fixed pair weight is rational in bulk."""
    s = 1/ratio
    sites = geometry.BOUNDARY
    outside = set().union(*(fields.neighbours(x) for x in sites)) - set(geometry.INTERIOR)
    knights = sorted(x for x in outside if sorted(map(abs, x)) == [0,1,2])
    endpoints = [[sites.index(y) for y in sorted(fields.neighbours(x).intersection(sites))]
                 for x in knights]
    assert len(endpoints) == 24 and all(len(edge) == 2 for edge in endpoints)
    axis = [max(map(abs,x)) == 2 for x in sites]
    assert all(axis[i] != axis[j] for i,j in endpoints)
    degrees = [sum(i in edge for edge in endpoints) for i in range(len(sites))]
    assert degrees == [4 if axial else 2 for axial in axis]
    assert all(not fields.neighbours(x).intersection(knights) for x in knights)
    A, B, C = (s**j + s**(6-j) for j in range(3))
    for m in range(5):
        plus = (s**m - s**(8-m))/(1-s**8)
        minus = (s**(4-m) - s**(4+m))/(1-s**8)
        assert min(plus, minus) >= 0
        vec = (A,B,C)
        assert all(plus*vec[j] + minus*vec[2-j] == s**(m+j) + s**(6-m-j)
                   for j in range(3))
    base = B**2/(A*C)
    source = C/A
    pairs = [(1 << i, 1 << j) for i,j in endpoints]
    powers = [base.numerator**j * base.denominator**(12-j) for j in range(13)]
    weights = []
    for w in words:
        disagreements = sum(bool(w & i) != bool(w & j) for i,j in pairs)
        assert disagreements % 2 == 0
        weights.append(powers[disagreements//2])
    # For nonuniform extreme sources, the sum of endpoint mismatches also has
    # even parity. Check the resulting exact factorization at selected words.
    for w in (words[0], words[9], words[len(words)//2], words[-1]):
        for flags in (0, 0x8c7001, 0xffffff):
            mismatches = []
            disagreements = 0
            for j,(i,k) in enumerate(pairs):
                pref = bool(flags & (1 << j))
                mismatches.append((bool(w & i) != pref) + (bool(w & k) != pref))
                disagreements += bool(w & i) != bool(w & k)
            n = sum(mismatches)
            assert n % 2 == disagreements % 2 == 0
            assert prod((A,B,C)[m]/A for m in mismatches) == base**(disagreements//2)*source**(n//2)
    # Each traced incident edge formerly contributed an extreme single-neighbour s.
    ratios = [v * source**(d//2)/s**d for v,d in zip(original_ratios,degrees)]
    assert all(0 < p < 1 for p in ratios)
    return z*np.asarray(weights, dtype=object), ratios


def corner_upper(r, values, z, owners, scale, ratios, verified, saved=None):
    """Certify one upper bound against all 14 corner-source classes and all shell words."""
    corners = list(product((-1,1), repeat=3))
    count = len(owners)
    configs = np.arange(count, dtype=np.int32)
    pc = np.fromiter((bin(i).count("1") for i in range(count)), dtype=np.int8)
    touched = [[i for i,y in enumerate(geometry.BOUNDARY) if y in fields.neighbours(x)]
               for x in corners]
    assert all(len(positions) == 3 for positions in touched)
    degrees = [sum(i in t for t in touched) for i in range(len(ratios))]
    assert degrees == [0 if max(map(abs,x)) == 2 else 2 for x in geometry.BOUNDARY]
    reduced = [v*r**d for v,d in zip(ratios,degrees)]
    counts = [pc[configs & sum(1 << i for i in group)] for group in touched]
    a,b = r.numerator,r.denominator
    trace = [a**(6-j)*b**j + a**j*b**(6-j) for j in range(4)]
    classes = fields.corner_classes(corners)
    assert len(classes) == 14 and all(F(0)<v<=1 for v in reduced)

    def factors(base, table, source):
        f = base.copy()
        for j,tally in enumerate(counts):
            f *= table[3-tally if source & (1 << j) else tally]
        return f

    def smooth(x):
        out = x.copy()
        for j,v in enumerate(reduced):
            width = 1 << j
            pairs = out.reshape(-1,2*width)
            left,right = pairs[:,:width].copy(),pairs[:,width:].copy()
            pairs[:,:width] = left+float(v)*right
            pairs[:,width:] = right+float(v)*left
        return out

    if saved is None:
        peak = max(z)
        normalized = np.asarray([n/peak for n in z], dtype=float)[owners]
        table = np.asarray([n/trace[0] for n in trace], dtype=float)
        signed = np.asarray([n/scale for n in values], dtype=float)[owners]
        greatest = -np.inf
        for source in classes:
            w = factors(normalized, table, source)
            test = smooth(w*signed)/smooth(w)
            assert np.isfinite(test).all()
            greatest = max(greatest, float(test.max()))
        # Floating point only selects a rational test. The exact sign is checked below.
        saved = min(verified, F(ceil(greatest*10**9)+5,10**9))
    else:
        assert saved <= verified
    coeff = (int(saved.numerator)*int(scale) - int(saved.denominator)*values)[owners]
    exact = z[owners]*coeff
    exact_table = np.asarray(trace, dtype=object)
    for k,source in enumerate(classes,1):
        w = factors(exact, exact_table, source)
        assert fields.transform(w, reduced).min() >= 0, (source, saved)
        if k in (1,7,14):
            print(f"exact corner sources checked: {k}/14", flush=True)
    return saved


def rebuild(ratio, green, expanded=False):
    _, _, translations, inequalities, shell, covariances = fixed.EXTENDED_PROOFS[-1]
    if expanded:
        translations = [(i, 0) for i in range(418)]
        inequalities = list(enumerate([0] * len(fixed.TESTS)))
        shell = [(p, 0) for p in fixed.local.exterior_patterns(4)]
        covariances = [(p, 342, 0) for p in fixed.local.shared_bonds()]
    ball = fixed.BallMoments(ratio, 8)
    assert len(ball.words) == 3236 and len(ball.rows) == 418
    u = (ratio-1)/(ratio+1)
    bare = u + 1/u
    angular = fixed.phase.angular_rows(ball)
    features = fixed.phase.reflection_features(ball)
    terms = list(fixed.additional_rows(ball, ratio, shell, covariances))
    for idx, mult in inequalities:
        tag, vector = fixed.TESTS[idx]
        if tag == "extension":
            row = fixed.exact_reflection(vector, bare, features)
        else:
            row = fixed.phase.spectral_cut(fixed.phase.polynomial(tag, vector), angular, bare, green)
        terms.append((mult, *row))
    if expanded:
        d = bare.denominator
        trace = 4*d*ball.g1.astype(object) + (bare.numerator - 6*d)*fixed.TOTAL_SCALE
        terms.append((0, fixed.local.round_row(trace,d), 4*fixed.local.ROUND + fixed.local.GRID, F(0)))
        if ratio >= F(25, 16):
            lower = FREE_TARGET
        elif ratio >= F(39, 25):
            lower = FREE_39
        elif ratio >= F(31, 20):
            lower = FREE_31_20
        else:
            raise ValueError("expanded physical inputs are certified only for r>=31/20")
        for label, num in lower.items():
            shift = tuple(map(int, label))
            x = next(x for x in geometry.SITES if geometry.add(x, shift) in geometry.INDEX)
            value = ball.averaged(geometry.mask([x, geometry.add(x, shift)]))
            terms.append((0, value - fixed.TOTAL_SCALE*num//10**10, fixed.local.ROUND, F(0)))
    return ball, translations, terms


def local_input_rows(ball, ratio, z, owners):
    """Free correlations at 79/50 and two deleted-region reflection squares."""
    from verify_boundary_reflection_blocks import layout
    assert ratio >= FREE_RATIO
    for label, number in FREE_79.items():
        shift = tuple(map(int, label))
        x = next(x for x in geometry.SITES if geometry.add(x, shift) in geometry.INDEX)
        values = ball.averaged(geometry.mask([x, geometry.add(x, shift)]))
        lower = F(number, 10**10)
        if label == "100":
            # Raise only the distinguished bond and then the remaining bonds by GKS.
            t = (ratio - FREE_RATIO)/(ratio + FREE_RATIO)
            lower = (lower+t)/(1+t*lower)
        yield 0, values - int(lower*fixed.TOTAL_SCALE), fixed.local.ROUND, F(0)
    sizes = np.bincount(owners, minlength=len(z))
    for kind, frame, a, b, indices in [
        ("axis", 11, 0, 7, (13, 121, 329)),
        ("diagonal", 0, 1, 64, (12, 220, 12)),
    ]:
        left, right, center = layout(kind)
        r = owners[center[frame] | left[[a,b],None] | right[None,[a,b]]]
        assert r.tolist() == [[indices[0],indices[1]],[indices[1],indices[2]]]
        coeff = {}
        for i, weight in zip(indices, (1, -2, 1)):
            coeff[i] = coeff.get(i, F(0)) + F(weight, int(sizes[i])*int(z[i]))
        # Positive rescaling is immaterial. Only three entries need rounding.
        norm = max(map(abs, coeff.values()))
        row = np.zeros(len(z), dtype=np.int64)
        for i, value in coeff.items():
            row[i] = round(fixed.TOTAL_SCALE*1024*value/norm)
        yield 0, row, fixed.local.GRID, F(0)


def residual(ball, translations, terms):
    denominator = fixed.DENOMINATOR
    values = np.zeros(len(ball.words), dtype=object)
    radius = F(sum(abs(a) for _, a in translations), denominator * fixed.SCALE)
    zero_mode = F(0)
    for idx, a in translations:
        values -= a * ball.rows[idx].astype(object)
    for b, row, rounding, mass in terms:
        assert b >= 0 and mass >= 0
        values += b * row.astype(object)
        radius += F(b, denominator)*rounding
        zero_mode += F(b, denominator)*mass
    assert zero_mode > 0
    return values, radius, zero_mode


def discovery_kernel(words, owners, inner, ratios):
    """Floating conditional probabilities on orbits. Used only to propose a dual."""
    bits = np.arange(len(owners), dtype=np.int32)
    lookup = np.ones(len(owners))
    for j, weight in enumerate(ratios):
        lookup[(bits & (1 << j)) != 0] *= float(weight)
    largest = max(inner)
    inner = np.asarray([v / largest for v in inner], dtype=float)
    kernel = np.empty((len(words), len(words)))
    owners = owners.astype(np.intp)
    for j, word in enumerate(words):
        hist = np.bincount(owners, weights=lookup[bits ^ word], minlength=len(words))
        weighted = inner * hist
        kernel[j] = weighted / np.sum(weighted)
    return kernel


def propose(ball, inherited, terms, kernel, bound):
    """Optimize a field-averaged lower bound using bounded continuous multipliers."""
    eq_indexes = [i for i, _ in inherited]
    matrix = np.stack([ball.rows[i] for i in eq_indexes] + [r for _, r, _, _ in terms]).astype(float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        projected = kernel @ (matrix / fixed.TOTAL_SCALE).T
    assert np.isfinite(projected).all()
    n = len(eq_indexes)
    if n > 150:
        _, _, pivot = qr(projected[:, :n], mode="economic", pivoting=True, check_finite=False)
        selected = list(map(int, pivot[:150]))
        projected = np.column_stack((projected[:, selected], projected[:, n:]))
        eq_indexes = [eq_indexes[j] for j in selected]
        n = len(selected)
    projected[:, :n] *= -1
    # The intercept dominates each field vector. The two sign choices for
    # each equality multiplier include its rigorous absolute rounding cost.
    e = 1.0/fixed.SCALE
    block = np.column_stack((projected[:, :n]+e, -projected[:, :n]+e,
                             projected[:, n:]+np.array([float(rad) for _, _, rad, _ in terms]),
                             -np.ones(len(kernel))))
    mass = np.r_[np.zeros(2*n), [float(q) for _, _, _, q in terms], 0.]
    cost = np.zeros(len(mass))
    cost[-1] = 1
    selected = np.linspace(0, len(block)-1, 250, dtype=int)
    best = None
    for _ in range(50):
        sol = linprog(cost, A_ub=block[selected], b_ub=np.zeros(len(selected)),
                      A_eq=[mass], b_eq=[1.], bounds=[(0, bound)]*(len(mass)-1)+[(None,None)],
                      method="highs-ipm", options={"time_limit": 90.})
        if not sol.success:
            if best is None:
                raise RuntimeError(sol.message)
            print(f"discovery stopped: {sol.message}; certifying the best prior iterate", flush=True)
            break
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            violated = block @ sol.x
        assert np.isfinite(violated).all()
        worst = np.argsort(-violated)[:125]
        attained = sol.fun+violated[worst[0]]
        if best is None or attained < best[0]:
            best = (attained, sol.x.copy())
        print(f"discovery {len(selected)} fields: objective={sol.fun:.10f}, "
              f"maximum violation={violated[worst[0]]:.4g}", flush=True)
        if violated[worst[0]] < 1.e-7 or attained < -0.000015:
            break
        added = np.setdiff1d(worst[violated[worst] > 1.e-8], selected)
        if not len(added):
            break
        selected = np.union1d(selected, added)
    assert best is not None
    x = best[1]
    den = fixed.DENOMINATOR
    translations = [(i, round(den*z)) for i, z in zip(eq_indexes, x[:n]-x[n:2*n])
                    if round(den*z)]
    weights = [max(0, round(den*z)) for z in x[2*n:-1]]
    return translations, weights
