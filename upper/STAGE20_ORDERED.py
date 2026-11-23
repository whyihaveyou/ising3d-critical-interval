"""Unfreeze unused physical rows in the zero-mode certificate on the 150-bond comparison graph."""

import argparse
import json
from fractions import Fraction
from math import floor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import verify_stage18_ordered_spectral_20261121 as previous


def add_new_tests(model, green):
    fixed = previous.W.fixed
    _, _, _, tests, _, _ = fixed.EXTENDED_PROOFS[-1]
    assert len(model.terms) == 47 and len(fixed.TESTS) == 48
    included = {idx for idx, _ in tests}
    angular = fixed.phase.angular_rows(model.ball)
    reflection = fixed.phase.reflection_features(model.ball)
    u = (model.ratio - 1) / (model.ratio + 1)
    bare = u + 1 / u
    result = {}
    for idx, (tag, root) in enumerate(fixed.TESTS):
        if idx in included:
            continue
        if tag == "extension":
            row = fixed.exact_reflection(root, bare, reflection)
        else:
            row = fixed.phase.spectral_cut(fixed.phase.polynomial(tag, root), angular, bare, green)
        result[len(model.terms)] = idx
        model.terms.append((0, *row))
    return result


def pool_for(model):
    original, _ = previous.source()
    active = {i for i, _ in original["ordinary"]}
    ordinary = [*original["ordinary"], *[[i, 0] for i in range(len(model.terms)) if i not in active]]
    return {**original, "ordinary": ordinary}, active


def discover(model, pool, active, layout, rounds):
    groups, specs, factors, masses = previous.parameters(model, pool)
    # Unused rows start at zero. The rescaling only changes LP search coordinates.
    for j, (name, key, _) in enumerate(specs, 1):
        if name == "ordinary" and key[0] not in active:
            step = 250000 if key[0] >= 28 else 25000
            groups[j][name][0][-1] = step
            specs[j-1] = (name, key, step)
            masses[j] = float(step * model.terms[key[0]][3] / previous.W.DEN)
            assert factors[j] == 0
    envelope = previous.prior.FieldEnvelope(model, groups, layout)
    best = None
    predicted = -np.inf
    for iteration in range(rounds+1):
        upper = envelope.separate(factors)
        print(f"round {iteration}, fields {len(envelope.constraints)}, U~{upper:+.12g}", flush=True)
        if best is None or upper < best[0]:
            best = upper, factors.copy()
        if upper < -0.00005 or upper <= predicted + 5.e-9 or iteration == rounds:
            break
        rows = np.asarray(list(envelope.constraints.values())) * 1e5
        n = len(specs)
        fit = linprog([0.] * n + [1.],
                      A_ub=np.column_stack((rows[:, 1:], -np.ones(len(rows)))),
                      b_ub=-rows[:, 0], A_eq=np.r_[masses[1:], 0.][None, :],
                      b_eq=[float(previous.MASS)], bounds=[(0., 8.)]*n + [(None, None)],
                      method="highs")
        assert fit.success, fit.message
        predicted = fit.x[-1] / 1e5
        factors = np.r_[1., fit.x[:-1]]
    return previous.integerise(model, specs, best[1][1:])


def check(args):
    assert 1 < args.ratio < Fraction(1242, 785)
    green = previous.W.fixed.phase.AngularGreen(Fraction(0))
    model = previous.W.Certificate(args.ratio, "order", green, False,
                                  bridges=True, expanded=False, corners=True)
    extra = add_new_tests(model, green)
    pool, base = pool_for(model)
    layout = previous.corner_geometry(model)
    path = Path(__file__).with_name(f"STAGE20_ORDERED_{args.ratio.numerator}_{args.ratio.denominator}.json")
    if args.mode == "discover":
        record, den = discover(model, pool, base, layout, args.rounds)
    else:
        saved = json.loads(path.read_text())
        assert saved["ratio"] == str(args.ratio)
        record, den = saved["rows"], saved["denominator"]
    assert type(den) is int and den > 0
    for name, rows in record.items():
        allowed = {tuple(entry[:-1]) for entry in pool[name]}
        used = [tuple(entry[:-1]) for entry in rows]
        assert len(used) == len(set(used)) and set(used) <= allowed
        assert all(type(w) is int for entry in rows for w in entry)
        assert all(entry[-1] != 0 if name == "translations" else entry[-1] > 0 for entry in rows)
    values, radius, q = previous.build_residual(model, record)
    radius /= den
    q /= den
    assert q == previous.MASS > 0
    scale = previous.W.DEN * previous.W.T * den
    if args.mode == "discover":
        cap, (source, field) = previous.prior.bridge.propose(model, values, layout, scale)
    else:
        cap = Fraction(saved["cap"])
        source, field = saved["witness"]
    assert source in layout[0] and 0 <= field < len(model.owners)
    weights = previous.corner_weights(model, source, layout[1], layout[3])
    weights = previous.direct_field(weights, layout[2], layout[4], layout[5], field)
    one = Fraction(int(weights @ values[model.owners]), int(weights.sum()) * scale)
    assert 0 <= cap - one < Fraction(1, 10**7)
    lower = -(cap + radius) / q
    active_extra = [[i, extra[i]] for i, _ in record["ordinary"] if i in extra]
    result = {"ratio": str(args.ratio), "denominator": den, "rows": record, "cap": str(cap),
              "radius": str(radius), "Q": str(q), "U": str(cap+radius), "L": str(lower),
              "one_plus_radius": str(one+radius), "witness": [source, field],
              "supplementary_test_ids": active_extra}
    print(f"r={args.ratio}, C={cap}, floor(1e15*L)={floor(10**15*lower)}, "
          f"floor(1e15*(one+radius))={floor(10**15*(one+radius))}, "
          f"field=({source},{field}), new ordinary indices={active_extra}", flush=True)
    if lower <= 0:
        assert args.mode in ("discover", "witness") and one + radius > 0
        print("PASS positive single-field obstruction for this candidate", flush=True)
    else:
        assert args.mode in ("discover", "directed", "raw")
        checker = (previous.W.base.corner_upper if args.mode == "raw"
                   else previous.CornerCheck(previous.W.base).all_corners)
        coarse = Fraction(max(map(int, values)), scale)
        assert checker(args.ratio, values, model.weights, model.owners, scale, model.ratios,
                       coarse, cap) == cap
        assert len(layout[0]) * len(model.owners) == 3670016
        print(f"PASS {args.mode} 3670016 fields, L={lower}", flush=True)
    if args.mode == "discover":
        path.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert result == saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratio", type=Fraction)
    parser.add_argument("mode", choices=("discover", "directed", "raw", "witness"))
    parser.add_argument("--rounds", type=int, default=48)
    assert __debug__
    check(parser.parse_args())
