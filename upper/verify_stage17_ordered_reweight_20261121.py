"""Reweight groups of frozen 136-record multipliers against the actual corner fields.

Discovery uses floating optimisation only. Replay reads the integer multipliers
and rational cap and certifies every extreme field, with either integer kernel.
"""
import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
from math import floor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import verify_ordered_136_bridge_corners_20261121 as bridge
from threshold_corner_compact_20261121 import average_float, CornerCheck
from verify_threshold_certificate_parameterization_20261121 import (
    build_residual, corner_geometry, corner_weights, direct_field,
)

W = bridge.W
HERE = Path(__file__).resolve().parent
TABLE = HERE / "stage17_ordered_reweight_20261121.json"
START = F(1394, 881)


def grouped_rows(model, original, individual):
    # Keep the six spectral multipliers fixed, hence Q stays fixed. Reflective
    # ordinary rows each get one factor; Walsh masks and signed translates are
    # grouped without modifying their allowed signs.
    groups = {}
    _, _, _, tests, shell, _ = W.fixed.EXTENDED_PROOFS[-1]
    for family, entries in original.items():
        for entry in entries:
            if family == "ordinary":
                global_test = tests[entry[0]-len(shell)][0]
                kind, _ = W.fixed.TESTS[global_test]
                tag = f"reflection {entry[0]}" if kind == "extension" else "fixed"
                assert tag == "fixed" or model.terms[entry[0]][3] == 0
            elif family == "walsh":
                tag = f"toggle {entry[1]}"
            else:
                tag = "positive translations" if entry[-1] > 0 else "negative translations"
            if individual and tag != "fixed":
                tag = family + " " + " ".join(str(k) for k in entry[:-1])
            group = groups.setdefault(tag, {key: [] for key in original})
            group[family].append(entry)
    assert len(groups["fixed"]["ordinary"]) == 6
    assert len(groups) == (236 if individual else 15)
    names = ["fixed"] + sorted(tag for tag in groups if tag != "fixed")
    return [groups[tag] for tag in names]


class FieldEnvelope:
    def __init__(self, model, groups, geometry):
        self.owners = model.owners
        self.words = geometry[4]
        self.ratios = geometry[2]
        self.weights = {}
        self.den = {}
        max_weight = max(model.weights)
        base = np.array([w/max_weight for w in model.weights])[self.owners]
        trace = np.array([v/geometry[3][0] for v in geometry[3]], dtype=float)
        for c in geometry[0]:
            x = base.copy()
            for j, count in enumerate(geometry[1]):
                x *= trace[3-count if c & (1 << j) else count]
            self.weights[c] = x
            self.den[c] = average_float(x, self.ratios)
        self.kernel = np.ones(len(self.words))
        for j, t in enumerate(self.ratios):
            self.kernel[(self.words & (1 << j)) != 0] *= float(t)
        orbits, error = [], []
        for group in groups:
            values, eps, _ = build_residual(model, group)
            orbits.append(np.asarray(values, dtype=float) / (W.DEN * W.T))
            error.append(float(eps))
        self.features = np.asarray(orbits).T
        self.errors = np.array(error)
        self.constraints = {}

    def separate(self, factors):
        orbit_values = np.einsum("ij,j->i", self.features, factors)[self.owners]
        rad = np.sum(self.errors*factors)
        worst = -np.inf
        for c, weights in self.weights.items():
            raw = average_float(weights*orbit_values, self.ratios)/self.den[c]
            assert np.isfinite(raw).all()
            worst = max(worst, float(raw.max()) + rad)
            # Add distinct near-maximal field values for each fixed corner source.
            distinct = set()
            for f in np.argpartition(-raw, 24)[:24]:
                key = c, int(f)
                value = round(float(raw[f]), 12)
                if value in distinct or key in self.constraints:
                    continue
                distinct.add(value)
                measure = weights * self.kernel[self.words ^ f]
                density = np.bincount(self.owners, weights=measure,
                                      minlength=len(self.features)) / np.sum(measure)
                features = np.einsum("i,ij->j", density, self.features) + self.errors
                assert np.isfinite(features).all()
                assert abs(np.dot(features, factors) - raw[f] - rad) < 1.e-9
                self.constraints[key] = features
        return worst

    def optimise(self, rounds):
        factors = np.ones(len(self.errors))
        best = None
        predicted = -np.inf
        for step in range(rounds + 1):
            upper = self.separate(factors)
            print(f"discovery step={step} fields={len(self.constraints)} U~{upper:+.12g}",
                  flush=True)
            if best is None or upper < best[0]:
                best = upper, factors.copy()
            if upper <= predicted+1.e-8 or step == rounds:
                break
            rows = np.asarray(list(self.constraints.values())) * 1.e5
            n = rows.shape[1]-1
            # t is U in units of 10^-5. The six fixed spectral rows supply
            # the affine column. Every adjustable multiplier keeps its sign.
            result = linprog([0]*n+[1],
                             A_ub=np.column_stack((rows[:, 1:], -np.ones(len(rows)))),
                             b_ub=-rows[:, 0], bounds=[(.55, 1.45)]*n+[(None, None)],
                             method="highs")
            assert result.success, result.message
            predicted = float(result.x[-1])/1.e5
            factors = np.r_[1., result.x[:-1]]
        return best


def adjusted(groups, factors):
    final = {key: [] for key in groups[0]}
    for group, factor in zip(groups, factors):
        for family, entries in group.items():
            for entry in entries:
                weight = round(entry[-1] * float(factor))
                assert weight * entry[-1] > 0
                final[family].append([*entry[:-1], weight])
    for entries in final.values():
        entries.sort(key=lambda e: tuple(e[:-1]))
    return final


def multipliers_digest(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def audit_source_field(model, original, geom):
    # A positive signed residual on just this all-aligned input suffices to
    # rule out a negative all-field cap for the unmodified source record.
    values, radius, Q = build_residual(model, original)
    weights = corner_weights(model, 0, geom[1], geom[3])
    density = direct_field(weights, geom[2], geom[4], geom[5], 0)
    w = F(int(density @ values[model.owners]), int(density.sum())*W.DEN*W.T) + radius
    print(f"old (c=0,f=0) residual plus radius: floor(10^15 R)={floor(10**15*w)}; "
          f"Q={Q}", flush=True)
    return w


def run(args):
    archive = json.loads(TABLE.read_text()) if TABLE.exists() else {}
    r = args.ratio or (F(1106, 699) if args.discover else min(map(F, archive)))
    assert 1 < r < START
    original = bridge.old.audited_record()
    green = W.fixed.phase.AngularGreen(F(0))
    model = W.Certificate(r, "order", green, False, bridges=True, expanded=False, corners=True)
    assert len(model.terms) == 47 and len(model.words) == 3236
    geom = corner_geometry(model)
    if args.source_field:
        assert audit_source_field(model, original, geom) > 0
        return
    if args.discover:
        mode = "individual" if args.individual else "fourteen"
    else:
        mode = archive[str(r)]["proposal_groups"]
    groups = grouped_rows(model, original, mode == "individual")
    if args.discover:
        env = FieldEnvelope(model, groups, geom)
        _, factors = env.optimise(args.rounds)
        record = adjusted(groups, factors)
        del env
    else:
        stored = archive[str(r)]
        record = stored["multipliers"]
        assert multipliers_digest(record) == stored["multiplier_sha256"]
    for family in record:
        reference = {tuple(row[:-1]): row[-1] for row in original[family]}
        weights = {tuple(row[:-1]): row[-1] for row in record[family]}
        assert weights.keys() == reference.keys() and len(weights) == len(record[family])
        assert all(weights[k]*reference[k] > 0 for k in weights)
    values, radius, Q = build_residual(model, record)
    assert Q == bridge.old.MASS > 0
    scale = W.DEN*W.T
    if args.discover:
        cap, (c, f) = bridge.propose(model, values, geom, scale)
    else:
        cap = F(stored["corner_cap"])
        c, f = stored["witness"]
    U, L = cap+radius, -(cap+radius)/Q
    w = corner_weights(model, c, geom[1], geom[3])
    density = direct_field(w, geom[2], geom[4], geom[5], f)
    total = int(density.sum())
    assert total > 0
    raw = F(int(density @ values[model.owners]), total*scale)
    assert 0 <= cap-raw < F(1, 10**7)
    changed = {}
    for family in record:
        reference = {tuple(row[:-1]): row[-1] for row in original[family]}
        changed[family] = sum(w[-1] != reference[tuple(w[:-1])] for w in record[family])
    print(f"r={r} changed={changed} U~{float(U):+.14g} L~{float(L):+.14g} "
          f"candidate source={c} field={f}", flush=True)
    if L <= 0:
        assert args.discover
        print("candidate gives no ordered certificate")
        return
    coarse = F(max(map(int, values)), scale)
    if args.raw_integers:
        verified = W.base.corner_upper(r, values, model.weights, model.owners,
                                       scale, model.ratios, coarse, cap)
    else:
        verified = CornerCheck(W.base).all_corners(r, values, model.weights,
                                                    model.owners, scale, model.ratios, coarse, cap)
    assert verified == cap
    output = dict(ratio=str(r), bridges=True, expanded=False, corners=True, proposal_groups=mode,
                  source_multiplier_sha256=bridge.old.SOURCE_HASH,
                  multiplier_sha256=multipliers_digest(record), multipliers=record,
                  changed=changed, corner_cap=str(cap), rounding_radius=str(radius),
                  Q=str(Q), U=str(U), L=str(L), floor_1e15_L=floor(10**15*L),
                  witness=[c, f], witness_residual=str(raw),
                  toggle_moments=bridge.comparison_inputs(model, record),
                  checked_fields=len(geom[0])*len(model.owners))
    assert output["checked_fields"] == 3670016
    if args.discover:
        if str(r) in archive and F(archive[str(r)]["L"]) >= L:
            print("existing point already has at least this lower bound", flush=True)
            return
        archive[str(r)] = output
        pending = TABLE.with_suffix(".pending")
        pending.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n")
        pending.replace(TABLE)
    else:
        assert output == stored
    print(f"PASS: all extreme fields, floor(10^15 L)={output['floor_1e15_L']}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=F, help="defaults to the smallest certified ratio on replay")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--individual", action="store_true",
                        help="let the 235 non-spectral multipliers vary independently during discovery")
    parser.add_argument("--raw-integers", action="store_true")
    parser.add_argument("--source-field", action="store_true",
                        help="test one exact obstruction for the unmodified old multipliers")
    options = parser.parse_args()
    assert __debug__
    run(options)
