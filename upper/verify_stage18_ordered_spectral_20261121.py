"""Optimise the six spectral multipliers with exactly the historical positive Q."""
import argparse
from fractions import Fraction as F
import json
from math import floor, gcd
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import qr

import verify_stage17_ordered_reweight_20261121 as prior
from threshold_corner_compact_20261121 import CornerCheck
from verify_threshold_certificate_parameterization_20261121 import (
    build_residual, corner_geometry, corner_weights, direct_field,
)

W = prior.W
HERE = Path(__file__).resolve().parent
STAGE17 = F(356, 225)
TABLE = HERE / "stage18_ordered_spectral_20261121.json"
MASS = prior.bridge.old.MASS
PIVOT = 29
SUBGRID = 1000


def source():
    data = json.loads(prior.TABLE.read_text())[str(STAGE17)]
    record = data["multipliers"]
    assert prior.multipliers_digest(record) == data["multiplier_sha256"]
    return record, data["multiplier_sha256"]


def empty_record():
    return {"translations": [], "ordinary": [], "walsh": []}


def parameters(model, record):
    # Nonnegative variables allow both signs on a translation. All comparison
    # and spectral multipliers remain nonnegative. Precondition columns using
    # their source weights; the smallest Walsh weights get some extra room.
    groups, specification, initial, masses = [empty_record()], [], [1.], [0.]
    for family, rows in record.items():
        for entry in rows:
            key = entry[:-1]
            seed = entry[-1]
            threshold = 50 if family == "walsh" else 500
            if family == "translations" and seed == 0:
                threshold = 30000
            scale = max(abs(seed), threshold)
            for sign in ((1, -1) if family == "translations" else (1,)):
                single = empty_record()
                step = scale*sign
                single[family].append([*key, step])
                groups.append(single)
                specification.append((family, tuple(key), step))
                initial.append(max(0., seed/step))
                slope = model.terms[key[0]][3] if family == "ordinary" else 0
                masses.append(float(step*slope / W.DEN))
    assert len(specification) == (2*len(record["translations"])+len(record["ordinary"])+len(record["walsh"]))
    assert abs(np.dot(masses, initial)-float(MASS)) < 1.e-12
    return groups, specification, np.array(initial), np.array(masses)


def extra_translations(model, record, geom, count):
    """Pick complementary translation coordinates on sampled physical fields."""
    selected = [i for i, _ in record["translations"]]
    missing = sorted(set(range(len(model.ball.rows))) - set(selected))
    assert 0 < count <= len(missing)
    envelope = prior.FieldEnvelope(model, [empty_record()], geom)
    rng = np.random.default_rng(20261121)
    words = geom[4]
    fields = [0, (1 << 18)-1, 15600, 188957]
    fields += [1 << j for j in range(18)]
    densities = []
    for source, weights in envelope.weights.items():
        sampled = [*fields, *rng.integers(len(words), size=25)]
        for field in sampled:
            w = weights*envelope.kernel[words ^ int(field)]
            densities.append(np.bincount(model.owners, weights=w,
                                         minlength=len(model.words)) / np.sum(w))
    samples = np.einsum("ki,ij->kj", np.asarray(densities), model.ball.rows.T / W.T)
    assert np.isfinite(samples).all() and np.max(abs(samples)) < 2.00001
    basis, singular, _ = np.linalg.svd(samples[:, selected], full_matrices=False)
    basis = basis[:, singular > 1.e-9*singular[0]]
    projected = np.einsum("ki,kj->ij", basis, samples[:, missing])
    remainder = samples[:, missing] - np.einsum("ki,ij->kj", basis, projected)
    assert np.isfinite(remainder).all()
    _, _, pivots = qr(remainder, mode="economic", pivoting=True, check_finite=False)
    return sorted(missing[i] for i in pivots[:count])


def optimise(model, record, geom, rounds):
    groups, spec, factors, masses = parameters(model, record)
    envelope = prior.FieldEnvelope(model, groups, geom)
    best = None
    predicted = -np.inf
    for step in range(rounds+1):
        upper = envelope.separate(factors)
        print(f"step={step} separated={len(envelope.constraints)} U~{upper:+.11g}", flush=True)
        if best is None or upper < best[0]:
            best = upper, factors.copy()
        if step == rounds or upper <= predicted + 5.e-9:
            break
        if step >= 5 and upper < -0.00035:
            break
        rows = np.asarray(list(envelope.constraints.values()))*1.e5
        n = len(spec)
        result = linprog([0.]*n+[1.],
                         A_ub=np.column_stack((rows[:, 1:], -np.ones(len(rows)))),
                         b_ub=-rows[:, 0],
                         A_eq=np.r_[masses[1:], 0.][None, :], b_eq=[float(MASS)],
                         bounds=[(0., 4.)]*n+[(None, None)], method="highs")
        assert result.success, result.message
        predicted = float(result.x[-1])/1.e5
        factors = np.r_[1., result.x[:-1]]
    return integerise(model, spec, best[1][1:])


def integerise(model, spec, solution):
    weights = {name: {} for name in empty_record()}
    for (name, idx, coeff), v in zip(spec, solution):
        weights[name][idx] = weights[name].get(idx, 0.) + float(v)*coeff
    aligned = {name: {idx: F(round(w*SUBGRID), SUBGRID) for idx, w in entries.items()}
               for name, entries in weights.items()}
    ordinary = aligned["ordinary"]
    assert model.terms[PIVOT][3] == F(1)
    # At a vertex of the LP the unit-slope row may have weight zero. Round
    # positive-mass nonpivot rows down first. If a floating equality happened
    # to overshoot by more than this truncation, remove its exact excess from
    # the largest positive contribution before recovering the pivot.
    for idx in ordinary:
        if idx != (PIVOT,) and model.terms[idx[0]][3] > 0:
            ordinary[idx] = F(floor(weights["ordinary"][idx]*SUBGRID), SUBGRID)
    positive = {idx: model.terms[idx[0]][3] for idx in ordinary
                if idx != (PIVOT,) and model.terms[idx[0]][3] > 0}
    pivot_weight = W.DEN*MASS - sum(slope*ordinary[idx] for idx, slope in positive.items())
    if pivot_weight < 0:
        largest = max(positive, key=lambda idx: positive[idx]*ordinary[idx])
        ordinary[largest] += pivot_weight / positive[largest]
        pivot_weight = F(0)
    ordinary[(PIVOT,)] = pivot_weight
    assert all(w >= 0 for name in ("ordinary", "walsh") for w in aligned[name].values())
    denominator = 1
    for entries in aligned.values():
        for w in entries.values():
            denominator = denominator*w.denominator // gcd(denominator, w.denominator)
    integers = {name: [[*idx, int(denominator*w)] for idx, w in sorted(entries.items()) if w]
                for name, entries in aligned.items()}
    assert sum(F(w, denominator*W.DEN)*model.terms[i][3]
               for i, w in integers["ordinary"]) == MASS
    return integers, denominator


def verify_domain(model, old, record, denominator):
    assert isinstance(denominator, int) and denominator > 0
    for family, entries in record.items():
        allowed = {tuple(row[:-1]) for row in old[family]}
        indexes = [tuple(row[:-1]) for row in entries]
        assert len(set(indexes)) == len(indexes) and set(indexes) <= allowed
        for e in entries:
            assert all(type(i) == int for i in e) and e[-1] != 0
            if family != "translations":
                assert e[-1] > 0
    og = dict(old["ordinary"])
    new = dict(record["ordinary"])
    _, _, _, tests, shell, _ = W.fixed.EXTENDED_PROOFS[-1]
    audit = {}
    for i in og:
        global_index = tests[i-len(shell)][0]
        tag, _ = W.fixed.TESTS[global_index]
        if tag != "extension":
            audit[str(i)] = {
                "global_test": global_index, "slope": str(model.terms[i][3]),
                "source_weight": og[i], "new_weight": str(F(new.get(i, 0), denominator)),
            }
    assert len(audit) == 6
    return audit


def run(args):
    table = json.loads(TABLE.read_text()) if TABLE.exists() else {}
    r = args.ratio or (F(791, 500) if args.discover else min(map(F, table)))
    assert 1 < r < STAGE17
    original, digest = source()
    green = W.fixed.phase.AngularGreen(F(0))
    model = W.Certificate(r, "order", green, False, bridges=True, expanded=False, corners=True)
    geom = corner_geometry(model)
    if args.discover:
        added = (extra_translations(model, original, geom, args.extra_translations)
                 if args.extra_translations else [])
    else:
        added = table[str(r)].get("extra_translation_indices", [])
    assert len(set(added)) == len(added)
    assert all(type(i) == int and 0 <= i < len(model.ball.rows) for i in added)
    assert set(added).isdisjoint(i for i, _ in original["translations"])
    pool = {**original, "translations": [*original["translations"], *[[i, 0] for i in added]]}
    if args.discover:
        record, denominator = optimise(model, pool, geom, args.rounds)
    else:
        stored = table[str(r)]
        record, denominator = stored["multipliers"], stored["multiplier_denominator"]
        assert prior.multipliers_digest(record) == stored["multiplier_sha256"]
    audit = verify_domain(model, pool, record, denominator)
    V, radius_integer, Q_integer = build_residual(model, record)
    assert len(geom[0]) == 14 and len(model.owners) == 2**18
    Q = Q_integer / denominator
    radius = radius_integer / denominator
    assert Q == MASS > 0
    scale = denominator*W.DEN*W.T
    if args.discover:
        cap, (c, f) = prior.bridge.propose(model, V, geom, scale)
    else:
        cap = F(stored["corner_cap"])
        c, f = stored["witness"]
    U = cap+radius
    lower = -U/Q
    weights = corner_weights(model, c, geom[1], geom[3])
    contracted = direct_field(weights, geom[2], geom[4], geom[5], f)
    den = int(contracted.sum())
    assert den > 0
    one = F(int(contracted @ V[model.owners]), den*scale)
    assert 0 <= cap-one < F(1, 10**7)
    active = {name: len(v) for name, v in record.items()}
    print(f"candidate r={r}, scaled Q={Q}, L~{float(lower):+.13g}, rows={active}, witness={(c, f)}",
          flush=True)
    if lower <= 0:
        assert args.discover
        if args.dump_negative:
            failure = {
                "ratio": str(r), "source": str(STAGE17), "source_hash": digest,
                "multiplier_sha256": prior.multipliers_digest(record),
                "multipliers": record, "multiplier_denominator": denominator,
                "spectral_rows": audit, "active_rows": active,
                "corner_cap": str(cap), "rounding_radius": str(radius),
                "Q": str(Q), "U": str(U), "L": str(lower),
                "floor_1e15_L": floor(10**15*lower),
                "witness": [c, f], "witness_raw_residual": str(one),
                "witness_plus_radius": str(one + radius),
                "classes": 14, "fields_per_class": 2**18,
                "moments": prior.bridge.comparison_inputs(model, original),
            }
            if added:
                failure["extra_translation_indices"] = added
            path = HERE / f"threshold_band_negative_{r.numerator}_{r.denominator}_20261121.json"
            path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
            print(f"wrote negative-candidate audit {path}", flush=True)
        return
    coarse = F(max(map(int, V)), scale)
    if args.raw_integers:
        assert W.base.corner_upper(r, V, model.weights, model.owners, scale,
                                    model.ratios, coarse, cap) == cap
    else:
        assert CornerCheck(W.base).all_corners(r, V, model.weights, model.owners, scale,
                                               model.ratios, coarse, cap) == cap
    out = {
        "ratio": str(r), "source": str(STAGE17), "source_hash": digest,
        "multiplier_sha256": prior.multipliers_digest(record),
        "multipliers": record, "multiplier_denominator": denominator,
        "spectral_rows": audit, "active_rows": active, "corner_cap": str(cap),
        "rounding_radius": str(radius), "Q": str(Q), "U": str(U), "L": str(lower),
        "floor_1e15_L": floor(10**15*lower), "witness": [c, f],
        "witness_raw_residual": str(one), "classes": 14, "fields_per_class": 2**18,
        "moments": prior.bridge.comparison_inputs(model, original),
    }
    if added:
        out["extra_translation_indices"] = added
    if args.discover:
        if str(r) in table and F(table[str(r)]["L"]) >= lower:
            return
        table[str(r)] = out
        pending = TABLE.with_suffix(".pending")
        pending.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
        pending.replace(TABLE)
    else:
        assert out == stored
    print(f"PASS floor(10^15 L)={out['floor_1e15_L']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=F)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--rounds", type=int, default=28)
    parser.add_argument("--raw-integers", action="store_true")
    parser.add_argument("--extra-translations", type=int, default=0,
                        help="select this many new geometric equalities during discovery only")
    parser.add_argument("--dump-negative", action="store_true",
                        help="save a nonpositive discovery candidate outside the positive archive")
    options = parser.parse_args()
    assert __debug__
    run(options)
